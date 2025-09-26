import os
import torch
import torch.nn as nn
import logging
import sys
from models.unet import UNet
from models.unetpp import UNetpp
from models.unetppp import UNetppp
from models.attentionunet import AttUNet
from models.avnet import AVNet
from models.csnet import CSNet
from models.ResUnet import ResUnet
import numpy as np
import utils
import seaborn as sns

from options.test_options import TestOptions
import cv2
import natsort
from BatchDataReader import OCTADataset
import scipy.misc as misc
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
os.environ['CUDA_VISIBLE_DEVICES'] = 'mps'

def test_net(net,device):

    test_dataset = OCTADataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_results = os.path.join(opt.saveroot, 'test_results')
    BGR = np.zeros((opt.data_size[0],opt.data_size[1],3))
    BGR_anno = np.zeros((opt.data_size[0], opt.data_size[1], 3))
    net.eval()

    #test set
    val_miou_sum = 0
    val_mDice_sum = 0
    val_siou = np.zeros((4, 1))
    val_num = opt.test_ids[1] - opt.test_ids[0]

    for itr, (test_images, test_annotations,cubename) in enumerate(test_loader):
        print('test: %d/%d'%(itr, len(test_loader)))
        img = np.zeros((opt.data_size[0], opt.data_size[1]))
        img[:, :] = test_images[0, 1, :, :]

        test_images = test_images.to(device=device, dtype=torch.float32)
        pred = net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        result = np.squeeze(pred_argmax).cpu().detach().numpy()
        test_annotations = test_annotations.cpu().detach().numpy()

        # save_path = './pred_result/' + cubename[0]
        # cv2.imwrite(save_path, result)

        heatmap = torch.softmax(pred, dim=1)
        heatmap = torch.max(heatmap, dim=1)
        heatmap = heatmap[0].cpu().detach().numpy()
        heatmap = 2 * (1 - heatmap[0])
        heatmap[test_annotations[0] < 2] = 0
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # cv2.imshow('heatmap', heatmap)
        # cv2.waitKey(10)

        miou_sum, ious = utils.cal_miou(result, test_annotations)
        val_miou_sum += miou_sum
        val_siou += ious
        val_mDice_sum += utils.cal_mDice(result, test_annotations)

        # print(cubename[0])
        diff_1 = pred_argmax - test_annotations
        diff_1[np.where(diff_1 != 0)] = 255
        diff_1[test_annotations<1] = 0
        diff = np.zeros((opt.data_size[0], opt.data_size[1], 3))
        diff[:, :, 0] = diff_1[0]
        diff[:, :, 1] = diff_1[0]
        diff[:, :, 2] = diff_1[0]

        # d[:, :, 1] = diff_2[0]
        # d[:, :, 2] = diff[0]

        # GRAY = np.zeros((opt.data_size[0], opt.data_size[1]))
        # GRAY[np.where(pred_argmax[0, :, :] == 1)] = 255
        # cv2.imwrite(os.path.join(test_results, cubename[0]), GRAY)
        a = np.zeros((opt.data_size[0], opt.data_size[1]))
        b = np.zeros((opt.data_size[0], opt.data_size[1]))
        c = np.zeros((opt.data_size[0], opt.data_size[1]))
        d = np.zeros((opt.data_size[0], opt.data_size[1]))
        e = np.zeros((opt.data_size[0], opt.data_size[1]))
        a[np.where(pred_argmax[0, :, :] == 0)] = 255 #background
        b[np.where(pred_argmax[0, :, :] == 1)] = 255 #mao xi xue guan
        c[np.where(pred_argmax[0, :, :] == 2)] = 255 #jing mai
        d[np.where(pred_argmax[0, :, :] == 3)] = 255 #dongmai
        e[np.where(pred_argmax[0, :, :] == 4)] = 255 #
        BGR[:, :, 0] = b #blue, maoxixueguan
        BGR[:, :, 1] = c #green
        BGR[:, :, 2] = d #red
        BGR[:, :, 2] += e #red

        a = np.zeros((opt.data_size[0], opt.data_size[1]))
        b = np.zeros((opt.data_size[0], opt.data_size[1]))
        c = np.zeros((opt.data_size[0], opt.data_size[1]))
        d = np.zeros((opt.data_size[0], opt.data_size[1]))
        e = np.zeros((opt.data_size[0], opt.data_size[1]))
        a[np.where(test_annotations[0, :, :] == 0)] = 255  # background
        b[np.where(test_annotations[0, :, :] == 1)] = 255  # mao xi xue guan
        c[np.where(test_annotations[0, :, :] == 2)] = 255  # jing mai
        d[np.where(test_annotations[0, :, :] == 3)] = 255  # dongmai
        e[np.where(test_annotations[0, :, :] == 4)] = 255  #
        BGR_anno[:, :, 0] = b  # blue, maoxixueguan
        BGR_anno[:, :, 1] = c  # green
        BGR_anno[:, :, 2] = d  # red
        BGR_anno[:, :, 2] += e  # red

        # cv2.imshow('original_img', img / 255)
        # cv2.waitKey(10)
        # cv2.imshow('diff', diff + heatmap * 0.8)
        # cv2.waitKey(10)
        # cv2.imshow('anno', BGR_anno)
        # cv2.waitKey(10)
        # cv2.imshow('prediction', BGR)
        # cv2.waitKey(10)
        # BGR[diff != 0] = diff[diff != 0]
        # cv2.imshow('segmentation', BGR)
        # cv2.waitKey(10)
        # cv2.imwrite(os.path.join(test_results, cubename[0]), BGR_anno)

    val_miou = val_miou_sum / val_num
    val_siou = val_siou / val_num
    val_mDice = val_mDice_sum / val_num
    print("Valid_mIoU:{}, Valid_mDice:{}".format(val_miou, val_mDice))
    print(val_siou)

if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f'Using device {device}')
    # loading network
    if opt.net_name == 'UNet':
        net = UNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    elif opt.net_name == 'AVNet':
        net = AVNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    elif opt.net_name == 'UNetpp':
        net = UNetpp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    elif opt.net_name == 'UNetppp':
        net = UNetppp(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    elif opt.net_name == 'AttUNet':
        net = AttUNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    elif opt.net_name == 'CSNet':
        net = CSNet(in_channels=opt.in_channels, n_classes=opt.n_classes)
    elif opt.net_name == 'ResUnet':
        net = ResUnet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    else:
        net = UNet(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #load trained model
    bestmodelpath = os.path.join(opt.saveroot, 'best_model',
                                 natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-4])
    restore_path = os.path.join(opt.saveroot, 'best_model',
                                natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-4]) + '/' + \
                   os.listdir(bestmodelpath)[0]
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
