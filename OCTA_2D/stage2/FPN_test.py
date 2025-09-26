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
import BatchDataReader

def test_net(net,device):
    test_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.test_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_results = os.path.join(opt.saveroot, 'test_results')
    net.eval()

    val_miou_sum = 0
    val_mDice_sum = 0
    val_ious = [0, 0, 0, 0]
    val_num = opt.test_ids[1] - opt.test_ids[0]

    for itr, (test_images, test_annotations, masks, original_preds, centerline_label, weight_map, cubename) in enumerate(test_loader):
        print(cubename)
        test_images = test_images.to(device=device, dtype=torch.float32)
        test_annotations = test_annotations.cpu().detach().numpy()
        original_preds = original_preds.cpu().detach().numpy()
        original_preds = original_preds[0]
        masks = masks.cpu().detach().numpy()
        _, pred = net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        result = np.squeeze(pred_argmax).cpu().detach().numpy()
        result[masks[0] < 1] = original_preds[masks[0] < 1]

        # save_path = './pred_result/second_stage/' + cubename[0]
        # cv2.imwrite(save_path, result)

        miou, ious = utils.cal_miou(result, test_annotations)
        val_miou_sum += miou
        val_ious += ious.squeeze()
        # val_miou_sum += utils.cal_miou(result, test_annotations)
        val_mDice_sum += utils.cal_mDice(result, test_annotations)
    val_miou = val_miou_sum / val_num
    val_mDice = val_mDice_sum / val_num
    val_ious = val_ious / val_num

    if val_num > 100:
        print("train_mIoU:{}, train_mDice:{}".format(val_miou, val_mDice))
        print(val_ious)
    else:
        print("Valid_mIoU:{}, Valid_mDice:{}".format(val_miou, val_mDice))
        print(val_ious)

    return val_miou, val_mDice, val_ious

if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    logging.info(f'Using device {device}')
    #loading network
    net = UNet_FPN(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    #load trained model
    bestmodelpath = os.path.join(opt.saveroot, 'best_model',
                                 natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-2])
    restore_path = os.path.join(opt.saveroot, 'best_model',
                                natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-2]) + '/' + \
                   os.listdir(bestmodelpath)[0]
    print(restore_path)
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
