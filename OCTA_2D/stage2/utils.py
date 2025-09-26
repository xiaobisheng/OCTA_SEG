import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

def cal_Dice(img1,img2):
    shape = img1.shape
    I = 0
    U = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] >= 1 and img2[i,j] >= 1:
                I += 1
            if img1[i,j] >= 1 or img2[i,j] >= 1:
                U += 1
    return 2*I/(I+U+1e-5)


def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])

def cal_miou(img1,img2):
    classnum = img2.max()
    iou=np.zeros((int(classnum),1))
    for i in range(int(classnum)):
        imga=img1==i+1
        imgb=img2==i+1
        imgi=imga * imgb
        imgu=imga + imgb
        iou[i]=np.sum(imgi)/np.sum(imgu)
    miou=np.mean(iou)
    return miou, iou

def cal_mDice(img1, img2):
    classnum = img2.max()
    Dice = np.zeros((int(classnum), 1))
    for i in range(int(classnum)):
        imga = img1 == i + 1
        imgb = img2 == i + 1
        imgi = imga * imgb
        Dice[i] = 2 * np.sum(imgi) / (np.sum(imga) + np.sum(imgb))
    mDice = np.mean(Dice)
    return mDice

def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    result = torch.zeros(shape)
    result.scatter_(1, input.cpu(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, masks):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()
        masks = masks.contiguous().view(target.shape[0], -1).cpu()
        predict[masks < 1] = target[masks < 1]

        # num = torch.zeros(target.shape[0])
        # den = torch.zeros(target.shape[0])
        #
        # for i in range(target.shape[0]):
        #     num[i] = torch.sum(torch.mul(predict[i, masks[i]>1], target[i, masks[i]>1])) + self.smooth
        #     den[i] = torch.sum(predict[i, masks[i]>1].pow(self.p) + target[i, masks[i]>1].pow(self.p)) + self.smooth

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight

    def forward(self, predict, target, masks, ignore_index=[]):
        shape = predict.shape
        target = torch.unsqueeze(target, 1)
        target = make_one_hot(target.long(), shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i not in ignore_index:
                dice_loss = dice(predict[:, i], target[:, i], masks)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / (target.shape[1] - len(ignore_index))

class CrossLoss(nn.Module):
    def __init__(self):
        super(CrossLoss, self).__init__()

    def forward(self, input_Artery, input_Vein, target_Artery, target_Vein, target_Artery_Skel, target_Vein_Skel, umge):
        return -cross_coef(input_Artery, input_Vein, target_Artery, target_Vein, target_Artery_Skel, target_Vein_Skel, umge)

def cross_coef(input_Artery, input_Vein, target_Artery, target_Vein, target_Artery_Skel, target_Vein_Skel,  umge):
    smooth = 1e-5
    target_cross = target_Artery_Skel * target_Vein_Skel
    conv = torch.ones([1, 1, 1 + 2 * umge, 1 + 2 * umge], requires_grad=True).cuda()
    input_Artery_conv = F.conv2d(input_Artery.unsqueeze(1), conv, padding=umge)
    input_Vein_conv = F.conv2d(input_Vein.unsqueeze(1), conv, padding=umge)
    target_Artery_conv = F.conv2d(target_Artery.unsqueeze(1).float(), conv, padding=umge)
    target_Vein_conv = F.conv2d(target_Vein.unsqueeze(1).float(), conv, padding=umge)
    return (target_cross.unsqueeze(1) * torch.abs(torch.abs(input_Artery_conv - target_Artery_conv) - torch.abs(input_Vein_conv - target_Vein_conv))).sum() / (target_cross.sum() + smooth)

class BranchLoss(nn.Module):
    def __init__(self):
        super(BranchLoss, self).__init__()

    def forward(self, input_Artery, input_Vein, target_Artery, target_Vein, branch_Artery, branch_Vein):
        return -branch_coef(input_Artery, input_Vein, target_Artery, target_Vein, branch_Artery, branch_Vein)

def branch_coef(input_Artery, input_Vein, target_Artery, target_Vein, branch_Artery, branch_Vein):
    smooth = 1e-5
    conv = torch.ones([1, 1, 3, 3], requires_grad=True).cuda()
    input_Artery_conv = F.conv2d(input_Artery.unsqueeze(1), conv, padding=1)
    input_Vein_conv = F.conv2d(input_Vein.unsqueeze(1), conv, padding=1)
    target_Artery_conv = F.conv2d(target_Artery.unsqueeze(1).float(), conv, padding=1)
    target_Vein_conv = F.conv2d(target_Vein.unsqueeze(1).float(), conv, padding=1)
    return ((branch_Artery.unsqueeze(1) * torch.abs(input_Artery_conv - target_Artery_conv)).sum() + (branch_Vein.unsqueeze(1) * torch.abs(input_Vein_conv - target_Vein_conv)).sum()) / (branch_Artery.sum() + branch_Vein.sum() + smooth)

# class CrossLossv2(nn.Module):
#     def __init__(self):
#         super(CrossLossv2, self).__init__()
#
#     def forward(self, input_Artery, input_Vein, target_Artery, target_Vein, target_Artery_Skel, target_Vein_Skel, umge):
#         return -cross_coef(input_Artery, input_Vein, target_Artery, target_Vein, target_Artery_Skel, target_Vein_Skel, umge)


