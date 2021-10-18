import torch
import torch.nn.functional as F
from torch import nn
from models.Transformer import TransformerModel
from models.Transformer import FixedPositionalEncoding


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, dilation=1, act='prelu'):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=dilation, bias=True,
                              dilation=dilation)

        if act == 'prelu':
            self.act = nn.PReLU(growthRate)
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        # (kernel_size - 1) // 2 + 1
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, dilation, act='***'):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, dilation=dilation, act=act))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class sub_pixel(nn.Module):
    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = []
        modules.append(nn.PixelShuffle(scale))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3, act='prelu'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        if act == 'prelu':
            self.act = nn.PReLU(growthRate)
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, act='***'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, act=act))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class Encoder(nn.Module):

    def __init__(self, channels=3):
        super(Encoder, self).__init__()

        nChannel = channels
        nDenselayer = 8
        nFeat = 32
        scale = 2
        growthRate = 16

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.down1 = nn.MaxPool2d(2)  # 384 - 192

        # dense layers
        self.dense1_1 = RDB(nFeat, nDenselayer, growthRate)
        self.dense1_2 = RDB(nFeat, nDenselayer, growthRate)
        self.d_ch1 = nn.Conv2d(nFeat, nFeat * 2, kernel_size=1, padding=0, bias=True)

        self.down2 = nn.MaxPool2d(2)  # 192 - 96

        self.dense2_1 = RDB(nFeat * 2, nDenselayer, growthRate)
        self.dense2_2 = RDB(nFeat * 2, nDenselayer, growthRate)
        self.d_ch2 = nn.Conv2d(nFeat * 2, nFeat * 4, kernel_size=1, padding=0, bias=True)

        self.down3 = nn.MaxPool2d(2)  # 96 - 48

        self.dense3_1 = RDB(nFeat * 4, nDenselayer, growthRate)
        self.dense3_2 = RDB(nFeat * 4, nDenselayer, growthRate)
        self.d_ch3 = nn.Conv2d(nFeat * 4, nFeat * 8, kernel_size=1, padding=0, bias=True)

        self.down4 = nn.MaxPool2d(2)  # 48 - 24


    def forward(self, input_tensor):
        feat1 = F.leaky_relu(self.conv1(input_tensor))
        feat2 = F.leaky_relu(self.conv2(feat1))

        # downsampling
        down1 = self.down1(feat2)

        # dense blocks
        dfeat1_1 = self.dense1_1(down1)
        dfeat1_2 = self.dense1_2(dfeat1_1)
        bdown1 = self.d_ch1(dfeat1_2)

        # downsampling
        down2 = self.down2(bdown1)

        dfeat2_1 = self.dense2_1(down2)
        dfeat2_2 = self.dense2_2(dfeat2_1)
        bdown2 = self.d_ch2(dfeat2_2)

        # downsampling
        down3 = self.down3(bdown2)

        dfeat3_1 = self.dense3_1(down3)
        dfeat3_2 = self.dense3_2(dfeat3_1)
        bdown3 = self.d_ch3(dfeat3_2)

        # downsampling
        down4 = self.down4(bdown3)

        return down4, torch.cat([dfeat3_2, dfeat3_1], 1),\
               torch.cat([dfeat2_2, dfeat2_1], 1),\
               torch.cat([dfeat1_2, dfeat1_1], 1), feat1


class GAB(nn.Module):

    def __init__(self):
        super(GAB, self).__init__()

        self.embedding = FixedPositionalEncoding()
        self.transformer = TransformerModel(256, 6, 8, 512)

    def forward(self, input_tensor):

        b, c, h, w = input_tensor.shape
        new_t = torch.reshape(input_tensor, [b, c, h * w])  # reshape image to sequence
        new_t = new_t.permute(0, 2, 1)  # permute axis to make it fully convolutional

        embed = self.embedding(new_t)
        transformed = self.transformer(embed)  # apply transformer
        transformed = transformed.permute(0, 2, 1)  # restore axis post transformer

        decoder_input = torch.reshape(transformed, [b, c, h, w])  # reshape to small image
        return decoder_input


class Decoder(nn.Module):

    def __init__(self, channels=3):
        super(Decoder, self).__init__()
        nChannel = channels

        nFeat = 32

        self.GFF_3x3_b = nn.Conv2d(nFeat * 8, nFeat * 8, kernel_size=3, padding=1, bias=True)

        # upsample
        self.up4 = nn.Upsample(scale_factor=2)  # 48 - 96

        self.GFF_1x1_5 = nn.Conv2d(nFeat * 8 * 2, nFeat * 8, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_5 = nn.Conv2d(nFeat * 8, nFeat * 4, kernel_size=3, padding=1, bias=True)

        # upsample
        self.up5 = nn.Upsample(scale_factor=2)  # 96 - 192
        self.GFF_1x1_6 = nn.Conv2d(nFeat * 4 * 2, nFeat * 4, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_6 = nn.Conv2d(nFeat * 4, nFeat * 2, kernel_size=3, padding=1, bias=True)

        self.up6 = nn.Upsample(scale_factor=2)  # 96 - 192
        self.GFF_1x1_7 = nn.Conv2d(nFeat * 2 * 2, nFeat * 2, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3_7 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)

        # upsample
        self.up7 = nn.Upsample(scale_factor=2)  # 192 - 384
        self.top_feat = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(nFeat, nChannel, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, decoder_input, down4, down3, down2, down1):

        bff_3 = self.GFF_3x3_b(decoder_input)
        # bottle neck dense dilated -------------------------------------------------------

        # upsample
        up4 = self.up4(bff_3)  # 96

        f_u4 = F.leaky_relu(torch.cat([up4, down4], 1))
        ff_up4_1 = self.GFF_1x1_5(f_u4)
        ff_up4_2 = F.leaky_relu(self.GFF_3x3_5(ff_up4_1))

        up5 = self.up5(ff_up4_2)  # 192

        f_u5 = torch.cat([up5, down3], 1)
        ff_up5_1 = F.leaky_relu(self.GFF_1x1_6(f_u5))
        ff_up5_2 = F.leaky_relu(self.GFF_3x3_6(ff_up5_1))

        up6 = self.up6(ff_up5_2)  # 384

        f_u6 = torch.cat([up6, down2], 1)
        ff_up6_1 = F.leaky_relu(self.GFF_1x1_7(f_u6))
        ff_up6_2 = F.leaky_relu(self.GFF_3x3_7(ff_up6_1))

        up7 = self.up7(ff_up6_2)  # 384
        final_cat = torch.cat([up7, down1], 1)
        top_f = F.leaky_relu(self.top_feat(final_cat))

        rgb = self.to_rgb(top_f)

        return rgb


class Network(nn.Module):

    def __init__(self, channels=3):
        super(Network, self).__init__()

        self.encoder = Encoder()
        self.GAB = GAB()
        self.decoder = Decoder()

    def forward(self, input_image):

        e_out, f4, f3, f2, f1 = self.encoder(input_image)
        attention_feat = self.GAB(e_out)
        rgb = self.decoder(attention_feat, f4, f3, f2, f1)

        return rgb, attention_feat


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    pool = Network()

    pool = nn.DataParallel(pool)
    pool.to('cuda:0')

    encoder = pool.module.encoder
    gab = pool.module.GAB
    # print(gab)

    print('# netRelighting parameters:', sum(param.numel() for param in pool.parameters()))
    # print(pool)
    out = pool(t)
    print(out[1].shape)