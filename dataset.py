import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import glob
import random



class ExposureCorrectionTrain(Dataset):

    def __init__(self, dataset_dir,
                 transform=None,
                 resize_size=(384, 384),
                 mode='train',
                 color=1):
        super(ExposureCorrectionTrain, self).__init__()

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.resize = resize_size

        # low light and normal light folders
        self.input_images = os.path.join(self.dataset_dir, 'INPUT_IMAGES')
        self.gt_images = os.path.join(self.dataset_dir, 'GT_IMAGES')

        if resize_size[0] > 384:
            self.image_list = read_and_parse(dataset_dir, resize_size[0])
        else:
            self.image_list = os.listdir(self.input_images)

        self.gt_dictionary = self.make_ground_truth_dictionary(self.gt_images)
        self.mode = mode

        self.color_mode = color

    def make_ground_truth_dictionary(self, gt_dir):

        gt_dictionary = {}

        files = os.listdir(os.path.join(self.dataset_dir, gt_dir))
        for i in range(len(files)):

            image_file = files[i]
            if image_file[-4:] != '.jpg':
                print(f'non image : {image_file}')

            image_index = image_file[:5]

            gt_dictionary[image_index] = image_file

        return gt_dictionary

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        input_image = self.image_list[index]
        image_prefix = input_image[:5]
        gt_name = self.gt_dictionary[image_prefix]

        input_path = os.path.join(self.input_images, input_image)
        image_contrast_path = glob.glob(os.path.join(self.input_images, f'*{image_prefix}*'))
        image_contrast_path.remove(input_path)
        image_contrast_path = image_contrast_path[random.randint(0, len(image_contrast_path) - 1)]
        gt_path = os.path.join(self.gt_images, gt_name)

        # read gt image ------------------------------------------------
        normal_image = load_image(gt_path, mode=self.color_mode)
        normal_image = torch.from_numpy(normal_image)
        normal_image = normal_image.permute(2, 0, 1)

        # read contrast image ------------------------------------------------
        contrast_image = load_image(image_contrast_path, mode=self.color_mode)
        contrast_image = torch.from_numpy(contrast_image)
        contrast_image = contrast_image.permute(2, 0, 1)

        # read input image ---------------------------------------------------
        input_image = load_image(input_path, mode=self.color_mode)
        input_image = torch.from_numpy(input_image)
        input_image = input_image.permute(2, 0, 1)

        # random crops on the images
        if self.mode == 'train':
            c, h, w = normal_image.shape
            i = np.random.randint(0, h - self.resize[1] + 1)
            j = np.random.randint(0, w - self.resize[0] + 1)
            normal_image = self._random_crop(normal_image, i, j)
            input_image = self._random_crop(input_image, i, j)
            contrast_image = self._random_crop(contrast_image, i, j)

        normalized_image = self.normalize_image(input_image)

        return normalized_image, normal_image, input_image, contrast_image

    def _random_crop(self, image, i=0, j=0):

        c, h, w = image.shape
        assert w >= self.resize[1] and h >= self.resize[0], \
            f'Error: Crop size: {self.resize[0]}, Image size: ({w}, {h})'

        PIL_image = transforms.functional.to_pil_image(image)
        cropped_image = transforms.functional.crop(PIL_image, i, j, self.resize[1], self.resize[0])
        cropped_image = transforms.functional.to_tensor(cropped_image)

        # nump = cropped_image.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(nump)
        # plt.show()

        return cropped_image

    def normalize_image(self, image):
        # normalize the image
        transform_list = [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        tr = transforms.Compose(transform_list)

        # normalized mage
        normalized_image = tr(image)

        return normalized_image

    def crop_image(self, image):

        pre = transforms.functional.to_pil_image(image)
        cropped = transforms.functional.center_crop(pre, self.resize[0])
        post = transforms.functional.to_tensor(cropped)

        # np = post.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(np)
        # plt.show()
        return post


class ExposureCorrectionTest(Dataset):

    def __init__(self, dataset_dir,
                 transform=None,
                 resize_size=(384, 384),
                 mode='train',
                 folder=None,
                 filt=3,
                 color=1):
        super(ExposureCorrectionTest, self).__init__()

        if folder is None:
            folder = ['INPUT_IMAGES', 'GT_IMAGES']

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.resize = resize_size

        # low light and normal light folders
        self.input_images = os.path.join(self.dataset_dir, folder[0])
        self.gt_images = os.path.join(self.dataset_dir, folder[1])

        self.image_list = os.listdir(self.input_images)
        self.gt_dictionary = self.make_ground_truth_dictionary(self.gt_images)

        self.image_list = self.filter_list(self.image_list, filt)
        self.mode = mode
        self.color_mode = color

    def make_ground_truth_dictionary(self, gt_dir):

        gt_dictionary = {}

        files = os.listdir(os.path.join(self.dataset_dir, gt_dir))
        for i in range(len(files)):

            image_file = files[i]
            if image_file[-4:] != '.jpg':
                print(f'non image : {image_file}')

            image_index = image_file[:5]

            gt_dictionary[image_index] = image_file

        return gt_dictionary

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        input_image = self.image_list[index]
        image_prefix = input_image[:5]
        gt_name = self.gt_dictionary[image_prefix]

        input_path = os.path.join(self.input_images, input_image)
        gt_path = os.path.join(self.gt_images, gt_name)

        # read gt image ------------------------------------------------

        normal_image = load_image(gt_path, mode=self.color_mode)
        nh, nw, nc = normal_image.shape

        if self.mode == 'test':
            ww, hh = adapt_size(nh, nw)
            ww, hh = get_novel_size(ww, hh, 512)
            normal_image = cv2.resize(normal_image, (ww, hh))

            # show_image(normal_image)

        normal_image = torch.from_numpy(normal_image)
        normal_image = normal_image.permute(2, 0, 1)

        # read input image ---------------------------------------------------
        input_image = load_image(input_path, mode=self.color_mode)
        if self.mode == 'test':
            input_image = cv2.resize(input_image, (ww, hh))
            # show_image(input_image)

        input_image = torch.from_numpy(input_image)
        input_image = input_image.permute(2, 0, 1)

        # random crops on the images
        if self.mode == 'train':
            c, h, w = normal_image.shape
            i = np.random.randint(0, h - self.resize[1] + 1)
            j = np.random.randint(0, w - self.resize[0] + 1)
            normal_image = self._random_crop(normal_image, i, j)
            input_image = self._random_crop(input_image, i, j)

        normalized_image = self.normalize_image(input_image)

        return normalized_image, normal_image, input_image

    def _random_crop(self, image, i=0, j=0):

        c, h, w = image.shape
        assert w >= self.resize[1] and h >= self.resize[0], \
            f'Error: Crop size: {self.resize[0]}, Image size: ({w}, {h})'

        PIL_image = transforms.functional.to_pil_image(image)
        cropped_image = transforms.functional.crop(PIL_image, i, j, self.resize[1], self.resize[0])
        cropped_image = transforms.functional.to_tensor(cropped_image)

        # nump = cropped_image.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(nump)
        # plt.show()

        return cropped_image

    def normalize_image(self, image):
        # normalize the image
        transform_list = [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        tr = transforms.Compose(transform_list)

        # normalized mage
        normalized_image = tr(image)

        return normalized_image

    def crop_image(self, image):

        pre = transforms.functional.to_pil_image(image)
        cropped = transforms.functional.center_crop(pre, self.resize[0])
        post = transforms.functional.to_tensor(cropped)

        # np = post.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(np)
        # plt.show()
        return post

    def filter_list(self, image_list, param):

        new_list = []
        if param == 1:
            for i in range(len(image_list)):
                a = image_list[i].split('_')[-1][0]
                if a == '0' or a == 'P':
                    new_list.append(image_list[i])

        elif param == 2:
            for i in range(len(image_list)):
                a = image_list[i].split('_')[-1][0]
                if a == 'N':
                    new_list.append(image_list[i])

        else:
            return image_list

        return new_list


class ExposureCorrection3(Dataset):

    def __init__(self, dataset_dir,
                 transform=None,
                 resize_size=(384, 384),
                 mode='train',
                 folder=None,
                 filt=3):
        super(ExposureCorrection3, self).__init__()

        if folder is None:
            folder = ['INPUT_IMAGES', 'GT_IMAGES']

        self.dataset_dir = dataset_dir
        self.transform = transform
        self.resize = resize_size

        # low light and normal light folders
        self.input_images = os.path.join(self.dataset_dir, folder[0])
        self.gt_images = os.path.join(self.dataset_dir, folder[1])

        self.image_list = os.listdir(self.input_images)
        self.gt_dictionary = self.make_ground_truth_dictionary(self.gt_images)

        self.image_list = self.filter_list(self.image_list, filt)
        self.mode = mode

    def make_ground_truth_dictionary(self, gt_dir):

        gt_dictionary = {}

        files = os.listdir(os.path.join(self.dataset_dir, gt_dir))
        for i in range(len(files)):

            image_file = files[i]
            if image_file[-4:] != '.jpg':
                print(f'non image : {image_file}')

            image_index = image_file[:5]

            gt_dictionary[image_index] = image_file

        return gt_dictionary

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        input_image = self.image_list[index]
        input_name = input_image
        image_prefix = input_image[:5]
        gt_name = self.gt_dictionary[image_prefix]

        input_path = os.path.join(self.input_images, input_image)
        gt_path = os.path.join(self.gt_images, gt_name)

        # read gt image ------------------------------------------------

        normal_image = load_image(gt_path)
        nh, nw, nc = normal_image.shape

        if self.mode == 'test':
            ww, hh = adapt_size(nh, nw)
            ww, hh = get_novel_size(ww, hh, 512)
            normal_image = cv2.resize(normal_image, (ww, hh))

            #show_image(normal_image)


        normal_image = torch.from_numpy(normal_image)
        normal_image = normal_image.permute(2, 0, 1)

        # read input image ---------------------------------------------------
        input_image = load_image(input_path)
        if self.mode == 'test':
            input_image = cv2.resize(input_image, (ww, hh))
            #show_image(input_image)

        input_image = torch.from_numpy(input_image)
        input_image = input_image.permute(2, 0, 1)

        # random crops on the images
        if self.mode == 'train':
            c, h, w = normal_image.shape
            i = np.random.randint(0, h - self.resize[1] + 1)
            j = np.random.randint(0, w - self.resize[0] + 1)
            normal_image = self._random_crop(normal_image, i, j)
            input_image = self._random_crop(input_image, i, j)

        normalized_image = self.normalize_image(input_image)


        return normalized_image, normal_image, input_image, input_name

    def _random_crop(self, image, i=0, j=0):

        c, h, w = image.shape
        assert w >= self.resize[1] and h >= self.resize[0], \
            f'Error: Crop size: {self.resize[0]}, Image size: ({w}, {h})'

        PIL_image = transforms.functional.to_pil_image(image)
        cropped_image = transforms.functional.crop(PIL_image, i, j, self.resize[1], self.resize[0])
        cropped_image = transforms.functional.to_tensor(cropped_image)

        # nump = cropped_image.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(nump)
        # plt.show()

        return cropped_image

    def normalize_image(self, image):
        # normalize the image
        transform_list = [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        tr = transforms.Compose(transform_list)

        # normalized mage
        normalized_image = tr(image)

        return normalized_image

    def crop_image(self, image):

        pre = transforms.functional.to_pil_image(image)
        cropped = transforms.functional.center_crop(pre, self.resize[0])
        post = transforms.functional.to_tensor(cropped)

        # np = post.permute(1, 2, 0).numpy()
        # plt.figure()
        # plt.imshow(np)
        # plt.show()
        return post

    def filter_list(self, image_list, param):

        new_list = []
        if param == 1:
            for i in range(len(image_list)):
                a = image_list[i].split('_')[-1][0]
                if a == '0' or a == 'P':
                    new_list.append(image_list[i])

        elif param == 2:
            for i in range(len(image_list)):
                a = image_list[i].split('_')[-1][0]
                if a == 'N':
                    new_list.append(image_list[i])

        else:
            return image_list

        return new_list


def get_novel_size(ww, hh, size):
    if ww > hh:
        ratio = size / ww
        nw, nh = round(ratio * ww), round(ratio * hh)
        return nw, nh
    else:
        ratio = size / hh
        nw, nh = round(ratio * ww), round(ratio * hh)
        return nw, nh


def load_image(name_jpg, mode=1):
    if mode == 1:
        return np.asarray(Image.open(name_jpg).convert('RGB')).astype(np.float32) / 255.0
    else:
        image = cv2.imread(name_jpg, cv2.IMREAD_COLOR)
        LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return LAB.astype(np.float32) / 255.0


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


def perform_test(h, size1, size2):
    if h > size1 and h < size2:
        return size1
    else:
        return 0


def adapt_size(h, w):
    nh = 0,
    nw = 0
    sizes = [64, 128, 256, 512, 1024, 2048, 5086]

    for i in range(len(sizes) - 1):
        nh = perform_test(h, sizes[i], sizes[i + 1])
        if nh != 0:
            break

    for i in range(len(sizes) - 1):
        nw = perform_test(w, sizes[i], sizes[i + 1])
        if nw != 0:
            break

    return nw, nh


def get_size_item():
    dataset_path = '/media/lf216/Data/elie/5k/data/INPUT_IMAGES'
    elements = os.listdir(dataset_path)
    list_element = []

    count = 0
    for image in elements:

        image_path = os.path.join(dataset_path, image)
        img = imageio.imread(image_path)
        H, W, C = img.shape

        if H > 768 and W > 768:
            count = count + 1
            list_element.append(image)

            with open("resolutions/images_768.txt", "a") as txt_file:
                txt_file.write(image + "\n")

            print(f'saved : {count}/{len(elements)} : {img.shape}')


def read_and_parse(file, res):
    f = f'images_{res}.txt'
    path = f'{file}/{f}'

    with open(path) as fs:
        lines = fs.readlines()

    lst = []
    for i in range(len(lines)):
        lst.append(lines[i].rstrip('\n'))

    return lst


if __name__ == '__main__':

    dataset = '/media/lf216/Data/elie/5k/data'
    path2 = '/media/lf216/Data/elie/5k/test'
    dat = ExposureCorrectionTrain(dataset)
    e = dat[8525]
    print(e[0].shape)

    # read_and_parse('/media/lf216/Data/elie/5k/data')
    # get_size_item()
