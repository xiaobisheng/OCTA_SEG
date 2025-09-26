# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
import torch
import torch.nn as nn

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=3, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(int(in_size/2), out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(out_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            # outputs0 = torch.cat([outputs0, input[i]], 1)
            outputs0 = outputs0 + input[i]
        return self.conv(outputs0)

class DOWN(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=2, padding=1):
        super(DOWN, self).__init__()
        self.maxpool1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(out_size),
                                      nn.ReLU(inplace=True), )

    def forward(self, inputs):
        return self.maxpool1(inputs)

class UNet_FPN(nn.Module):

    def __init__(self, in_channels, n_classes, channels=64, is_deconv=True, is_batchnorm=True):
        super(UNet_FPN, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.channels = channels
        self.n_classes = n_classes

        # downsampling
        self.conv1 = unetConv2(self.in_channels, self.channels, self.is_batchnorm)
        self.maxpool1 = DOWN(self.channels, self.channels)
        self.conv2 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool2 = DOWN(self.channels, self.channels)
        self.conv3 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool3 = DOWN(self.channels, self.channels)
        self.conv4 = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool4 = DOWN(self.channels, self.channels)
        self.center = unetConv2(self.channels, self.channels, self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.up_concat3 = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.up_concat2 = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.up_concat1 = unetUp(self.channels * 2, self.channels, self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)
        self.ACT = nn.ReLU()

        self.conv1_2nd = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool1_2nd = DOWN(self.channels, self.channels)
        self.conv2_2nd = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool2_2nd = DOWN(self.channels, self.channels)
        self.conv3_2nd = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool3_2nd = DOWN(self.channels, self.channels)
        self.conv4_2nd = unetConv2(self.channels, self.channels, self.is_batchnorm)
        self.maxpool4_2nd = DOWN(self.channels, self.channels)
        self.center_2nd = unetConv2(self.channels, self.channels, self.is_batchnorm)

        self.up_concat4_2nd = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.up_concat3_2nd = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.up_concat2_2nd = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.up_concat1_2nd = unetUp(self.channels * 2, self.channels, self.is_deconv)
        self.outconv2 = nn.Conv2d(self.channels, self.n_classes, 3, padding=1)
        self.ACT_2nd = nn.ReLU()



    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)

        up4 = self.up_concat4(center, conv4)
        up3 = self.up_concat3(up4, conv3)
        up2 = self.up_concat2(up3, conv2)
        up1 = self.up_concat1(up2, conv1)

        up = self.outconv1(up1)
        output = self.ACT(up)

        conv1_2nd = self.conv1_2nd(up1+conv1)
        maxpool1_2nd = self.maxpool1_2nd(conv1_2nd)
        conv2_2nd = self.conv2_2nd(maxpool1_2nd+conv2)
        maxpool2_2nd = self.maxpool2_2nd(conv2_2nd)
        conv3_2nd = self.conv3_2nd(maxpool2_2nd+conv3)
        maxpool3_2nd = self.maxpool3_2nd(conv3_2nd)
        conv4_2nd = self.conv4_2nd(maxpool3_2nd+conv4)
        maxpool4_2nd = self.maxpool4_2nd(conv4_2nd)
        center_2nd = self.center_2nd(maxpool4_2nd)

        up4_2nd = self.up_concat4(center_2nd, conv4_2nd)
        up3_2nd= self.up_concat3_2nd(up4_2nd, conv3_2nd)
        up2_2nd = self.up_concat2_2nd(up3_2nd, conv2_2nd)
        up1_2nd = self.up_concat1_2nd(up2_2nd, conv1_2nd)

        up_2nd = self.outconv2(up1_2nd)
        output2 = self.ACT_2nd(up_2nd)

        return output, output2
