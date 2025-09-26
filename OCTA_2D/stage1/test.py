import os
import torch
import torch.nn as nn
import logging
import sys
from models.unet_FPN import UNet_FPN
import numpy as np
import utils
from options.test_options import TestOptions
import cv2
import natsort
from BatchDataReader import OCTADataset

def test_net(net,device):
    test_dataset = OCTADataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    net.eval()

    val_miou_sum = 0
    val_mDice_sum = 0
    val_siou = np.zeros((4, 1))
    val_num = opt.test_ids[1] - opt.test_ids[0]
    for itr, (test_images, test_annotations,cubename) in enumerate(test_loader):
        print(cubename[0])
        img = np.zeros((opt.data_size[0], opt.data_size[1]))
        img[:, :] = test_images[0, 1, :, :]

        test_images = test_images.to(device=device, dtype=torch.float32)
        _, pred = net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        result = np.squeeze(pred_argmax).cpu().detach().numpy()
        test_annotations = test_annotations.cpu().detach().numpy()

        # save_path = './pred_result/' + cubename[0]
        # cv2.imwrite(save_path, result)

        miou_sum, ious = utils.cal_miou(result, test_annotations)
        val_miou_sum += miou_sum
        val_siou += ious
        val_mDice_sum += utils.cal_mDice(result, test_annotations)

        # result = np.array(result, dtype=np.uint8)
        # test_annotations = np.array(test_annotations, dtype=np.uint8)
        # cv2.imshow('pred', result * 60)
        # cv2.imshow('gt', test_annotations[0] * 60)
        # cv2.waitKey(1000)

    val_miou = val_miou_sum / val_num
    val_siou = val_siou / val_num
    val_mDice = val_mDice_sum / val_num
    print("Valid_mIoU:{}, Valid_mDice:{}".format(val_miou, val_mDice))
    print(val_siou)

if __name__ == '__main__':
    #setting logging and loading options
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    opt = TestOptions().parse()
    #setting GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #loading network
    net = UNet_FPN(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #load trained model
    bestmodelpath = os.path.join(opt.saveroot, 'best_model',
                                 natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-2])
    restore_path = os.path.join(opt.saveroot, 'best_model',
                                natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-2]) + '/' + \
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
