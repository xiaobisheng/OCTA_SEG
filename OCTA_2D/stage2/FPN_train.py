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
import BatchDataReader
import torch.nn.functional as F
from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler

def hamming_like_loss(pred_logits, gt_labels):
    """
    pred_logits: (N, 5, H, W) - predicted logits
    gt_labels: (N, H, W) - ground truth labels in {0,1,2,3,4}
    
    Return:
        Scalar tensor loss: each pixel gets loss=1 if top-1 prediction != GT, else 0
    """
    # Get top-1 predicted class (still allows gradients to flow from logits)
    pred_class = torch.argmax(pred_logits, dim=1)  # (N, H, W), non-differentiable wrt logits
    # To keep it differentiable, we instead build a pseudo-one-hot for argmax
    pred_onehot = F.one_hot(pred_class, num_classes=5).permute(0, 3, 1, 2).float()  # (N, 5, H, W)
    
    # Ground truth one-hot
    gt_onehot = F.one_hot(gt_labels, num_classes=5).permute(0, 3, 1, 2).float()  # (N, 5, H, W)

    # Hamming-like distance: 1 if incorrect, 0 if correct
    per_pixel_hamming = (pred_onehot - gt_onehot).abs().sum(dim=1) / 2  # (N, H, W)
    # Why divide by 2? Because one-hot diff will give 2 for wrong class, 0 for correct

    return per_pixel_hamming


def evaluate_model(valid_loader, val_num, epoch):
    val_miou_sum = 0
    val_mDice_sum = 0
    val_ious = [0, 0, 0, 0]
    net.eval()
    for itr, (test_images, test_annotations, masks, original_preds, centerline_label, weight_map, cubename) in enumerate(valid_loader):
        test_images = test_images.to(device=device, dtype=torch.float32)
        test_annotations = test_annotations.cpu().detach().numpy()
        original_preds = original_preds.cpu().detach().numpy()
        original_preds = original_preds[0]
        masks = masks.cpu().detach().numpy()
        _, pred = net(test_images)
        pred_argmax = torch.argmax(pred, dim=1)
        result = np.squeeze(pred_argmax).cpu().detach().numpy()
        result[masks[0] < 1] = original_preds[masks[0] < 1]
        miou, ious = utils.cal_miou(result, test_annotations)
        val_miou_sum += miou
        val_ious += ious.squeeze()
        # val_miou_sum += utils.cal_miou(result, test_annotations)
        val_mDice_sum += utils.cal_mDice(result, test_annotations)
    val_miou = val_miou_sum / val_num
    val_mDice = val_mDice_sum / val_num
    val_ious = val_ious / val_num

    if val_num > 100:
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
    train_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids,opt.data_size,opt.modality_filename,is_dataaug=True) #######
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    train_eval_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.train_ids, opt.data_size, opt.modality_filename,
                                                is_dataaug=False)
    train_eva_loader = torch.utils.data.DataLoader(train_eval_dataset, batch_size=1, shuffle=False, num_workers=4)#########
    valid_dataset = BatchDataReader.CubeDataset(opt.dataroot, opt.val_ids, opt.data_size, opt.modality_filename,is_dataaug=False)
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
        decay_steps=len(train_loader) * 120,
        total_steps=len(train_loader) * 200,
    )

    #Setting Loss
    Loss_CE = nn.CrossEntropyLoss(reduction='none')
    Loss_DSC = utils.DiceLoss()

    #Start train
    for epoch in range(1, opt.num_epochs + 1):
        print('epoch: %d' % (epoch))
        net.train()
        net.to(device=device)
        for itr, (train_images, train_annotations, masks, original_preds, centerline_label, weight_map, name) in enumerate(train_loader):
            train_images = train_images.to(device=device, dtype=torch.float32)
            train_annotations = train_annotations.to(device=device, dtype=torch.long)
            masks = masks.to(device=device, dtype=torch.long)
            weight_map = weight_map.to(device=device, dtype=torch.float32)
            centerline_label = centerline_label.to(device=device, dtype=torch.long)

            pred, pred2 = net(train_images)

            loss_ce = Loss_CE(pred, train_annotations)
            loss_ce = loss_ce[masks == 1]
            loss_ce = loss_ce.mean()
            loss_dice = Loss_DSC(pred, train_annotations, masks)

            loss_ce_2 = Loss_CE(pred2, train_annotations)
            loss_ce_2 = loss_ce_2[masks == 1]
            loss_ce_2 = loss_ce_2.mean()
            loss_dice_2= Loss_DSC(pred2, train_annotations, masks)

            loss_ce_3 = Loss_CE(pred, train_annotations)
            loss_ce_3 = loss_ce_3 * weight_map
            loss_ce_3 = loss_ce_3[weight_map > 0]
            loss_ce_3 = loss_ce_3.mean()

            # loss_ce_4 = Loss_CE(pred, centerline_label)
            # loss_ce_4 = loss_ce_4[centerline_label>1]
            # loss_ce_4 = loss_ce_4.mean()
            # loss_dice_4 = Loss_DSC(pred, centerline_label, masks, ignore_index=[0, 1])
            loss_hamming = hamming_like_loss(pred, train_annotations)
            loss_hamming = loss_hamming[centerline_label>1]
            loss_hamming = loss_hamming.mean()

            loss = loss_ce + 0.6 * loss_dice + loss_ce_2 + 0.6 * loss_dice_2 + 0.1 * (loss_ce_3 + loss_hamming)
            # loss = loss_ce.mean()
            if itr%20 == 5:
                print('iteration: %d, loss: %.6f, loss_ce1: %.6f, loss_dice1: %.6f, loss_ce_weight: %.6f, loss_hamming: %.6f, lr: %.6f' % 
                      (itr, loss.cpu().detach().numpy(), loss_ce.cpu().detach().numpy(), loss_dice.cpu().detach().numpy(),
                       loss_ce_3.cpu().detach().numpy(), loss_hamming.cpu().detach().numpy(),
                        optimizer.state_dict()['param_groups'][0]['lr']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Start Val
        with torch.no_grad():
            # Save model
            torch.save(net.module.state_dict(), os.path.join(model_save_path, f'{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved !')

            # Calculate validation mIOU
            train_miou, train_mDice, train_ious = evaluate_model(train_eva_loader, train_num, epoch)
            val_miou, val_mDice, val_ious = evaluate_model(valid_loader, val_num, epoch)

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




