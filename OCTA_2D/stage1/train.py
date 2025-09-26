import os
import torch
import torch.nn as nn
import numpy as np
import logging
import sys
from models.unet_FPN import UNet_FPN
import utils
import shutil
import natsort
from options.train_options import TrainOptions
from BatchDataReader import OCTADataset
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler

def evaluate_model(valid_loader, val_num, epoch, phase='train'):
    val_miou_sum = 0
    val_mDice_sum = 0
    val_ious = [0, 0, 0, 0]
    net.eval()
    # pbar = tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader))
    for itr, (test_images, test_annotations, cubename) in enumerate(valid_loader):
        test_images = test_images.to(device=device, dtype=torch.float32)
        test_annotations = test_annotations.cpu().detach().numpy()
        _, pred = net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        result = np.squeeze(pred_argmax).cpu().detach().numpy()
        miou, ious = utils.cal_miou(result, test_annotations)
        val_miou_sum += miou
        val_ious += ious.squeeze()
        # val_miou_sum += utils.cal_miou(result, test_annotations)
        val_mDice_sum += utils.cal_mDice(result, test_annotations)
    val_miou = val_miou_sum / val_num
    val_mDice = val_mDice_sum / val_num
    val_ious = val_ious / val_num

    if phase == 'train':
        print("Step:{}, train_mIoU:{}, train_mDice:{}".format(epoch, val_miou, val_mDice))
        print(val_ious)
    else:
        print("Step:{}, Valid_mIoU:{}, Valid_mDice:{}".format(epoch, val_miou, val_mDice))
        print(val_ious)

    return val_miou, val_mDice, val_ious


def train_net(net,device):
    #train setting
    val_num = opt.val_ids[1] - opt.val_ids[0]
    train_num = opt.train_ids[1] - opt.train_ids[0]
    best_valid_miou = 0
    model_save_path = os.path.join(opt.saveroot, 'checkpoints')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model')
    # Read Data
    train_dataset = OCTADataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,is_dataaug=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    train_eval_dataset = OCTADataset(opt.dataroot, opt.train_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    train_eva_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=1, shuffle=False, num_workers=4)
    valid_dataset = OCTADataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.999))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)

    scheduler = TriStageLRScheduler(
        optimizer,
        init_lr=0.00005,
        peak_lr=0.0005,
        final_lr=0.0001,
        init_lr_scale=1,
        final_lr_scale=0.1,
        warmup_steps=len(train_loader) * 30,
        hold_steps=len(train_loader) * 50,
        decay_steps=len(train_loader) * 220,
        total_steps=len(train_loader) * 300,
    )

    #Setting Loss
    Loss_CE = nn.CrossEntropyLoss()
    Loss_DSC = utils.DiceLoss()

    #Start train
    for epoch in range(1, opt.num_epochs + 1):
        net.train()
        net.to(device=device)
        for itr, (train_images, train_annotations, name) in enumerate(train_loader):
            train_images = train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            pred0, pred = net(train_images)

            loss = Loss_CE(pred0, train_annotations) + 0.6 * Loss_DSC(pred0, train_annotations) + Loss_CE(pred, train_annotations) + 0.6 * Loss_DSC(pred, train_annotations)
            if itr%20 == 5:
                print('%.6f, %.6f' % (loss.cpu().detach().numpy(), optimizer.state_dict()['param_groups'][0]['lr']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Start Val
        with torch.no_grad():
            # Calculate validation mIOU
            train_miou, train_mDice, train_ious = evaluate_model(train_eva_loader, train_num, epoch, phase='train')
            val_miou, val_mDice, val_ious = evaluate_model(valid_loader, val_num, epoch, phase='val')

            # Save model
            if epoch % 20 == 1 or val_miou > best_valid_miou:
                torch.save(net.module.state_dict(), os.path.join(model_save_path, f'{epoch}.pth'))
                logging.info(f'Checkpoint {epoch} saved !')

            # save best model
            if val_miou > best_valid_miou:
                temp = '{:.6f}'.format(val_miou)
                os.mkdir(os.path.join(best_model_save_path, temp))
                temp2 = f'{epoch}.pth'
                shutil.copy(os.path.join(model_save_path, temp2), os.path.join(best_model_save_path, temp, temp2))
                model_names = natsort.natsorted(os.listdir(best_model_save_path))
                if len(model_names) == 4:
                    shutil.rmtree(os.path.join(best_model_save_path, model_names[0]))
                best_valid_miou = val_miou


if __name__ == '__main__':
    # setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # loading options
    opt = TrainOptions().parse()#########
    # setting GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f'Using device {device}')
    #loading network
    net = UNet_FPN(in_channels=opt.in_channels, n_classes=opt.n_classes, channels=opt.channels)
    net = torch.nn.DataParallel(net, [0])
    # load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    # input the model into GPU
    # net.to(device=device)
    try:
        train_net(net=net,device=device) ############
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




