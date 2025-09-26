import os
import natsort
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import numpy as np
import cv2
import random
from skimage import transform
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.stats import linregress

def add_soft_weight_map(weight_map, point, radius, max_weight):
    """weight_map: 2D numpy, point: (x, y), radius: int, max_weight: float"""
    xx, yy = np.meshgrid(np.arange(weight_map.shape[0]), np.arange(weight_map.shape[1]), indexing='ij')
    distance = np.sqrt((xx - point[0])**2 + (yy - point[1])**2)
    # 高斯核，可调整sigma
    sigma = radius / 2
    kernel = np.exp(-0.5 * (distance / sigma) ** 2)
    kernel[distance > radius] = 0
    weight_map += kernel * max_weight
    return weight_map


class CubeDataset(Dataset):
    def __init__(self, data_dir, data_id, data_size, modality, is_dataaug=True):
        self.is_dataaug=is_dataaug
        self.datanum=data_id[1]-data_id[0]
        self.modality=modality
        self.data_size=data_size
        self.modalitynum=len(modality)-1
        self.datasetlist={'data':{},'label':{}, 'pred':{}}
        for modal in modality:
            if modal != modality[-1] and modal != self.modality[-2]:
                self.datasetlist['data'].update({modal: {}})
                imglist = os.listdir(os.path.join(data_dir, modal))
                imglist = natsort.natsorted(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['data'][modal].update({img: {}})
                    imgadress= os.path.join(data_dir, modal, img)
                    self.datasetlist['data'][modal][img] = imgadress
            elif modal == self.modality[-2]:
                imglist = os.listdir('./pred_result/best_model_unetAvgpooling')
                imglist = natsort.natsorted(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['pred'].update({img: {}})
                    labeladdress = os.path.join('./pred_result/best_model_unetAvgpooling', img)
                    self.datasetlist['pred'][img] = labeladdress
            else:
                imglist = os.listdir(os.path.join(data_dir, modal))
                imglist = natsort.natsorted(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['label'].update({img: {}})
                    labeladdress = os.path.join(data_dir, modal, img)
                    self.datasetlist['label'][img] = labeladdress

        # for i in range(300):
        #     self.__getitem__(i)

    def __getitem__(self, index):#index from where?
        data =np.zeros((self.modalitynum,self.data_size[0],self.data_size[1]))
        label = np.zeros((self.data_size[0],self.data_size[1]))
        pred = np.zeros((self.data_size[0],self.data_size[1]))
        mask = np.zeros((self.data_size[0],self.data_size[1]))

        for i,modal in enumerate(self.modality):
            if modal != self.modality[-1] and modal != self.modality[-2]:
            # if modal != self.modality[-1]:
                name=list(self.datasetlist['data'][modal])[index]
                data[i,:,:]= cv2.imread(self.datasetlist['data'][modal][name], cv2.IMREAD_GRAYSCALE).astype(np.float16)
                # cv2.imshow('img', cv2.imread(self.datasetlist['data'][modal][name], cv2.IMREAD_GRAYSCALE))
                # cv2.waitKey(2000)
            elif modal == self.modality[-2]:
                name = list(self.datasetlist['pred'])[index]
                pred[:, :] = cv2.imread(self.datasetlist['pred'][name], cv2.IMREAD_GRAYSCALE).astype(np.float16)
                # print(len(pred[pred==0]), len(pred[pred==1]), len(pred[pred==2]), len(pred[pred==3]), len(pred[pred==4]))

            else:
                name = list(self.datasetlist['label'])[index]
                label[:,:]=cv2.imread(self.datasetlist['label'][name], cv2.IMREAD_GRAYSCALE).astype(np.float16)

        #data augmentation
        if self.is_dataaug==True:
            data, label, pred = self.augmentation(data, label, pred)
            # pred = self.pixel_aug(pred, label)
            pred, weight_map = self.generate_gaussian_mask(label, pred)
        else:
            weight_map = np.zeros((self.data_size[0],self.data_size[1]))

        mask[pred > 1] = 1
        mask = cv2.dilate(mask, None)

        data[-1] = pred
        
        centerline_label = self.get_centerline(label)
        centerline_label = torch.from_numpy(np.ascontiguousarray(centerline_label))
        weight_map = torch.from_numpy(np.ascontiguousarray(weight_map))

        data = torch.from_numpy(np.ascontiguousarray(data))
        label = torch.from_numpy(np.ascontiguousarray(label))
        pred = torch.from_numpy(np.ascontiguousarray(pred))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        # label[label>1] = 0
        return data,label, mask, pred, centerline_label, weight_map, name

    def __len__(self):
        return self.datanum
    
    def generate_gaussian_mask(self, annotation, preds):
        label_copy = annotation.copy()
        label_copy[label_copy < 2] = 0
        label_copy[label_copy == 4] = 0
        label_copy[label_copy >= 2] = 1
        centerline = skeletonize(label_copy.astype(np.uint8))
        centerline = np.array(centerline, dtype=np.uint8)

        ends, branches = self.extract_keypoints(centerline)

        # 构建权重map
        weight_map = self.build_soft_weight_map(centerline, ends, branches)

        # 随机在关键点处扰乱map
        index = np.random.choice(len(branches), int(0.4 * len(branches)), replace=False)
        for i in index:
            num = random.randint(10, 40)
            patch = preds[branches[i][0] - num: branches[i][0] + num, branches[i][1] - num: branches[i][1] + num]
            if random.random() < 0.5:
                patch[patch==2] = 3
                preds[branches[i][0] - num: branches[i][0] + num, branches[i][1] - num: branches[i][1] + num] = patch
            else:
                patch[patch == 3] = 2
                preds[branches[i][0] - num: branches[i][0] + num, branches[i][1] - num: branches[i][1] + num] = patch
        
        index = np.random.choice(len(ends), int(0.4 * len(ends)), replace=False)
        for i in index:
            num = random.randint(10, 20)
            patch = preds[ends[i][0] - num: ends[i][0] + num, ends[i][1] - num: ends[i][1] + num]
            if random.random() < 0.5:
                patch[patch==1] = 2
                preds[ends[i][0] - num: ends[i][0] + num, ends[i][1] - num: ends[i][1] + num] = patch
            
        return preds, weight_map
    
    def build_soft_weight_map(self, centerline, end_centers, branch_centers,
                              end_radius=10, branch_radius=10,
                              end_weight=2.0, branch_weight=2.0, base_weight=1.0):
        """
        centerline: 0/1 numpy array
        end_centers/branch_centers: N x 2 ndarray（坐标列表）
        最终输出: float 权重图
        """
        weight_map = np.zeros_like(centerline, dtype=np.float32)
        # 加强端点周围
        for pt in end_centers:
            weight_map = add_soft_weight_map(weight_map, pt, end_radius, end_weight - base_weight)
        # 加强分叉周围
        for pt in branch_centers:
            weight_map = add_soft_weight_map(weight_map, pt, branch_radius, branch_weight - base_weight)
        return weight_map

    def extract_keypoints(self, centerline):
        """
            输入:
              centerline: (H, W) 二值numpy数组
            返回:
              ends: (N, 2) 端点坐标
              branches: (M, 2) 分叉点坐标
            """
        # 定义邻域核
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # 不计中心自己
        # 卷积得每点8邻域像素数
        neighbor_count = convolve(centerline, kernel, mode='constant', cval=0)
        # mask采样中心线点
        endpoints_mask = (centerline == 1) & (neighbor_count == 1)
        branchpoints_mask = (centerline == 1) & (neighbor_count >= 3)
        # 获取xy坐标
        ends = np.argwhere(endpoints_mask)
        branches = np.argwhere(branchpoints_mask)
        ends = self.deduplicate(ends)
        branches = self.deduplicate(branches)
        return ends, branches
    
    def deduplicate(self, pixel_):
        points = []
        points.append([pixel_[0][0], pixel_[0][1]])
        for i in range(len(pixel_)):
            candidate_poi = [pixel_[i][0], pixel_[i][1]]
            candidate_to_points = True
            for poi in points:
                if abs(candidate_poi[0] - poi[0]) + abs(candidate_poi[1] - poi[1]) < 10:
                    candidate_to_points = False
                    break
            if candidate_to_points:
                points.append(candidate_poi)
        return np.array(points)

    
    def get_centerline(self, label):
        centerline = np.zeros(label.shape)
        for i in range(1, int(np.max(label))+1):
            mask = np.zeros(label.shape)
            mask[label==i] = 1
            skeleton = skeletonize(mask)
            skeleton = np.array(skeleton, dtype=np.uint8)
            centerline[skeleton == 1] = i

        return centerline

    def find_endpoints(self, mask):
        skeleton = skeletonize(mask)
        skeleton = np.uint8(skeleton)

        # cv2.imshow('mask', skeleton * 255)
        # cv2.waitKey(2000)

        end_points = []
        h, w = skeleton.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x]:
                    neighbours = skeleton[y-1:y+2, x-1:x+2].sum()-1
                    if neighbours == 1:
                        end_points.append((x, y))

        skeleton = skeleton*255
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        # skeleton = self.extend_curve(end_points, skeleton)
        for (x, y) in end_points:
            cv2.circle(skeleton, (x, y), radius=10, color=(255, 255, 255), thickness=-1)

        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('mask', skeleton)
        # cv2.waitKey(2000)

        return skeleton

    def pixel_aug(self, pred, label):
        # 角点检测
        key_points = np.zeros((self.data_size[0],self.data_size[1]))
        label_copy = label.copy()
        label_copy[label_copy < 2] = 0
        gray = np.float32(label_copy)
        dst = cv2.cornerHarris(gray, 3, 3, 0.05)
        dst = cv2.erode(dst, None)
        dst = cv2.dilate(dst, None)
        key_points[dst > 0.06 * dst.max()] = 1

        # label_copy_ = label_copy.astype(np.uint8) * 30
        # label_copy_[key_points == 1] = 255
        # cv2.imshow('label', label_copy_)
        # cv2.imshow('mask', key_points)
        # cv2.waitKey(2000)

        # BGR = np.zeros((pred.shape[0], pred.shape[1], 3))
        # a = np.zeros((pred.shape[0], pred.shape[1]))
        # b = np.zeros((pred.shape[0], pred.shape[1]))
        # c = np.zeros((pred.shape[0], pred.shape[1]))
        # d = np.zeros((pred.shape[0], pred.shape[1]))
        # e = np.zeros((pred.shape[0], pred.shape[1]))
        # a[np.where(pred == 0)] = 255  # background
        # b[np.where(pred == 1)] = 255  # mao xi xue guan
        # c[np.where(pred == 2)] = 255  # jing mai
        # d[np.where(pred == 3)] = 255  # dongmai
        # e[np.where(pred == 4)] = 255  #
        # BGR[:, :, 0] = b  # blue, maoxixueguan
        # BGR[:, :, 1] = c  # green
        # BGR[:, :, 2] = d  # red
        # BGR[:, :, 2] += e  # red
        # cv2.imshow('Pred', BGR)
        # cv2.waitKey(2000)

        pixel = np.where(key_points == 1)
        pixel_ = []
        for i in range(len(pixel[0])):
            pixel_.append([pixel[0][i], pixel[1][i]])
        pixel = np.array(pixel_)
        points = []
        points.append([pixel_[0][0], pixel_[0][1]])

        for i in range(len(pixel)):
            candidate_poi = [pixel_[i][0], pixel_[i][1]]
            candidate_to_points = True
            for poi in points:
                if abs(candidate_poi[0] - poi[0]) + abs(candidate_poi[1] - poi[1]) < 8:
                    candidate_to_points = False
                    break
            if candidate_to_points:
                points.append(candidate_poi)


        index = np.random.choice(len(points), int(0.4 * len(points)), replace=False)
        for i in index:
            num = random.randint(10, 80)
            patch = pred[points[i][0] - num: points[i][0] + num, points[i][1] - num: points[i][1] + num]
            if random.random() < 0.5:
                patch[patch==2] = 3
                # patch[patch==3] = 2
                pred[points[i][0] - num: points[i][0] + num, points[i][1] - num: points[i][1] + num] = patch
            else:
                # patch[patch == 2] = 3
                patch[patch == 3] = 2
                pred[points[i][0] - num: points[i][0] + num, points[i][1] - num: points[i][1] + num] = patch

        return pred

    # image augmentation
    def cutmix(self, img, anno, pred):
        img1, img2 = img.copy(), img.copy()
        anno1, anno2 = anno.copy(), anno.copy()
        pred1, pred2 = pred.copy(), pred.copy()
        h1, h2 = anno1.shape[0], anno2.shape[0]
        w1, w2 = anno1.shape[1], anno2.shape[1]
        # 设定lamda的值，服从beta分布
        alpha = 1.0
        lam = np.random.beta(alpha, alpha)
        cut_rat = np.sqrt(1. - lam)
        # 裁剪第二张图片
        cut_w = int(w2 * cut_rat)  # 要裁剪的图片宽度
        cut_h = int(h2 * cut_rat)  # 要裁剪的图片高度
        # uniform
        cx = np.random.randint(w2)  # 随机裁剪位置
        cy = np.random.randint(h2)

        # 限制裁剪的坐标区域不超过2张图片大小的最小值
        xmin = np.clip(cx - cut_w // 2, 0, min(w1, w2))  # 左上角x
        ymin = np.clip(cy - cut_h // 2, 0, min(h1, h2))  # 左上角y
        xmax = np.clip(cx + cut_w // 2, 0, min(w1, w2))  # 右下角x
        ymax = np.clip(cy + cut_h // 2, 0, min(h1, h2))  # 右下角y

        img1[:, ymin:ymax, xmin:xmax] = img2[:, ymin:ymax, xmin:xmax]
        anno1[ymin:ymax, xmin:xmax] = anno2[ymin:ymax, xmin:xmax]
        pred1[ymin:ymax, xmin:xmax] = pred2[ymin:ymax, xmin:xmax]

        return img1, anno1, pred1


    def augmentation(self, image, annotation, pred):
        # rotate
        # if torch.randint(0, 4, (1,)) == 0:
        #     angle = torch.randint(-30,30, (1,))
        #     angle = random.randint(-30, 30)
        #     for i in range(self.modalitynum):
        #         image[i,:,:] = transform.rotate(image[i,:,:], angle)
        #     annotation= transform.rotate(annotation, angle)
        #     pred = transform.rotate(pred, angle)
        # flipud
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = np.flipud(image[i,:,:])
            annotation= np.flipud(annotation)
            pred = np.flipud(pred)
        #fliplr
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = np.fliplr(image[i,:,:])
            annotation = np.fliplr(annotation)
            pred = np.fliplr(pred)
            '''
        #noise
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = random_noise(image[i,:,:], mode='gaussian')
        '''
        return image, annotation, pred
