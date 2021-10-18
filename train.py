import os
import torch

gpus = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import shutil
import torch.optim as optim
import torch.utils.data
from torch import nn
from dataset import ExposureCorrectionTrain, ExposureCorrectionTest
import torchvision.utils as utils

from models.Network import Network as HDRNet
from pytorch_msssim import SSIM, MS_SSIM
from utils.metrics import PSNR

import json
from models.vgg19 import VGGLoss as VGG2
from torch.utils.tensorboard import SummaryWriter
from pytorch_ssim import SSIM as SSIM2

SSIM2_metric = SSIM2()


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 1 * (1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 1 * (1 - super(SSIM_Loss, self).forward(img1, img2))


class SSIM_Test(SSIM):
    def forward(self, img1, img2):
        return super(SSIM_Test, self).forward(img1, img2)


def preprocess_for_vgg(image):
    mean = torch.reshape(torch.tensor([0.485, 0.456, 0.406], device=image.device), (1, 3, 1, 1))
    std = torch.reshape(torch.tensor([0.229, 0.224, 0.225], device=image.device), (1, 3, 1, 1))
    input_tensor = (image - mean) / std
    return input_tensor


def create_or_recreate_folders(configs):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """

    folders = [configs['display_folder'],
               configs['summary'],
               configs['epoch_folder'],
               config['display_val']]

    # iterate through the folders and delete them if they exist
    # then recreate them.
    # otherwise simply create them
    for i in range(len(folders)):
        folder = folders[i]
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            os.mkdir(folder)


def load_config(file):
    """
    takes as input a file path and returns a configuration file
    that contains relevant information to the training of the NN
    :param file:
    :return:
    """

    # load the file as a raw file
    loaded_file = open(file)

    # conversion from json file to dictionary
    configuration = json.load(loaded_file)

    # returning the file to the caller
    return configuration


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_features(contrast_image, network):
    encoder_feat, _, _, _, _ = network.module.encoder(contrast_image)
    contrast_feat = network.module.GAB(encoder_feat)
    return contrast_feat


config = load_config('config.json')['config']
writer = SummaryWriter(config['summary'])
print(config)

display_folder = config['display_folder']
display_validation = config['display_val']
epoch_folder = config['epoch_folder']
train_mode = config['training']['mode']

# ------------------------- spliting -------------------------

if train_mode == 'train':
    dataset = ExposureCorrectionTrain(config['data_path'], resize_size=(128, 128))
    validation = ExposureCorrectionTest(config['val_path'], mode='test')

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               num_workers=config['data_workers'],
                                               batch_size=config['data_batch'],
                                               shuffle=True)

    validation_loader = torch.utils.data.DataLoader(validation,
                                                    num_workers=config['data_workers'],
                                                    batch_size=config['val_batch'],
                                                    shuffle=False)

    print(f'Training Length : {len(dataset)}')
    print(f'Validation Length : {len(validation)}')
# ------------------------- spliting -------------------------


# creating the network and others
network = HDRNet()
vgg = VGG2()

print('# Network parameters:', sum(param.numel() for param in network.parameters()))
opt = optim.Adam(network.parameters(), lr=config['training']['lr'])

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

network = nn.DataParallel(network)
network.to(device)

vgg = nn.DataParallel(vgg)
vgg.to(device)

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/Model{config['training']['epoch']}.pth", map_location='cuda')
    network.load_state_dict(weight)

    epoch = config['training']['epoch']
else:
    create_or_recreate_folders(config)
    epoch = 0

# setting up the loss function
psnr_metric = PSNR(max_value=1)
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
ssim_loss = SSIM()
l1_smooth = torch.nn.SmoothL1Loss()

# params settings
NUM_EPOCHS = config['training']['epochs']
display_iter = config['training']['display']

for epoch in range(1 + epoch, NUM_EPOCHS + 1):
    running_results = {'batch_sizes': 0, 'LossC_ILD': 0, 'LossC_ITD': 0, 'LossR_TFT': 0, 'Loss_SSIM': 0,
                       'Loss_perception': 0, 'Loss_grad': 0}
    # netRelighting.train()
    iteration = 0
    mse_loss_total = 0.0
    vgg_total = 0.0
    ssim_total = 0.0
    consistency_loss_total = 0.0
    network = network.train()
    for tensors in train_loader:

        input_image = tensors[0]
        gt_image = tensors[1]
        low_image = tensors[2]
        contrast_image = tensors[3]

        iteration += 1
        batch_size = gt_image.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # Update network
        ###########################
        if torch.cuda.is_available():
            input_image = input_image.to(device)
            gt_image = gt_image.to(device)
            low_image = low_image.to(device)
            contrast_image = contrast_image.to(device)

        network.zero_grad()

        # computing the output
        prediction_tensor, features = network(input_image)
        contrast_image_features = compute_features(contrast_image, network)

        ssim_ = SSIM2_metric(prediction_tensor, gt_image)

        loss_target_second_tensor = l1_smooth(prediction_tensor, gt_image)
        reconstruction_loss = loss_target_second_tensor
        mse_ = l2_loss(prediction_tensor, gt_image)

        # compute EC loss function
        consistency_loss = l1_loss(contrast_image_features, features)

        # computing vgg loss
        vgg_loss = vgg(preprocess_for_vgg(prediction_tensor), preprocess_for_vgg(gt_image))

        loss = reconstruction_loss + vgg_loss + 0.1 * consistency_loss
        loss.mean().backward()
        opt.step()

        print('[%d/%d][%d], L1 : %f, CL : %f, LR : %f' % (
            epoch, NUM_EPOCHS, iteration,
            loss_target_second_tensor.item(), consistency_loss.item(),
            get_lr(opt)))

        # every 500 iters finished, output results
        if iteration % display_iter == 0:
            with torch.no_grad():
                display_data = torch.cat(
                    [low_image,
                     prediction_tensor,
                     gt_image,
                     ], dim=0)

                utils.save_image(display_data, display_folder + "/Epoch_%d Iter_%d.jpg" % (epoch, iteration),
                                 nrow=batch_size, padding=2, normalize=False)

        mse_loss_total += mse_.item()
        ssim_total += ssim_.item()
        consistency_loss_total += consistency_loss.item()

    # ----------------------------------------------------------------------------------------------------------------------------
    scheduler.step()
    # schedulerD.step()
    writer.add_scalar('MSE', mse_loss_total / iteration, epoch)
    writer.add_scalar('ssim loss', ssim_total / iteration, epoch)

    # one epoch finished, output training loss, save models
    torch.save(network.state_dict(), epoch_folder + '/Model%d.pth' % epoch)

    # execute the network on the validation set
    iteration = 0
    l2_loss_total = 0.0
    vgg_total = 0.0
    ssim_total = 0.0
    psnr_total = 0.0
    network = network.eval()

    for tensors in validation_loader:

        input_image = tensors[0]
        gt_image = tensors[1]
        low_image = tensors[2]

        iteration += 1
        batch_size = gt_image.size(0)
        running_results['batch_sizes'] += batch_size

        if torch.cuda.is_available():
            input_image = input_image.to(device)
            gt_image = gt_image.to(device)
            low_image = low_image.to(device)

        with torch.no_grad():
            predicted_image, _ = network(input_image)

        print(f'Validation : {iteration}')
        display_data = torch.cat(
            [low_image,
             predicted_image,
             gt_image,
             ], dim=0)

        if epoch % 20 == 0:
            utils.save_image(display_data, display_validation + "/Epoch_%d Val_%d.jpg" % (epoch, iteration),
                             nrow=batch_size, padding=2, normalize=False)

        l2 = l2_loss(predicted_image, gt_image)
        psnr = psnr_metric(predicted_image, gt_image, max_value=1)
        loss_ssim = SSIM2_metric(predicted_image, gt_image)  # ssim_loss(predicted_image, gt_image)

        l2_loss_total += l2.item()
        psnr_total += psnr.item()
        ssim_total += loss_ssim.item()

    writer.add_scalar('MSE validation', l2_loss_total / iteration, epoch)
    writer.add_scalar('PSNR validation', psnr_total / iteration, epoch)
    writer.add_scalar('SSIM validation', ssim_total / iteration, epoch)
