import os
import natsort
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import cv2

class OCTADataset(Dataset):
    def __init__(self, data_dir,data_id,data_size,modality,is_dataaug=True):
        self.is_dataaug=is_dataaug
        self.datanum=data_id[1]-data_id[0]
        self.modality=modality
        self.data_size=data_size
        self.modalitynum=len(modality)-1
        self.datasetlist={'data':{},'label':{}}
        for modal in modality:
            if modal != modality[-1]:
                self.datasetlist['data'].update({modal: {}})
                imglist = os.listdir(os.path.join(data_dir, modal))
                imglist = natsort.natsorted(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['data'][modal].update({img: {}})
                    imgadress= os.path.join(data_dir, modal, img)
                    self.datasetlist['data'][modal][img] = imgadress
            else:
                imglist = os.listdir(os.path.join(data_dir, modal))
                imglist = natsort.natsorted(imglist)
                for img in imglist[data_id[0]:data_id[1]]:
                    self.datasetlist['label'].update({img: {}})
                    labeladdress = os.path.join(data_dir, modal, img)
                    self.datasetlist['label'][img] = labeladdress

    def __getitem__(self, index):#index from where?
        data =np.zeros((self.modalitynum,self.data_size[0],self.data_size[1]))
        label = np.zeros((self.data_size[0],self.data_size[1]))
        for i,modal in enumerate(self.modality):
            if modal != self.modality[-1]:
                name=list(self.datasetlist['data'][modal])[index]
                data[i,:,:]= cv2.imread(self.datasetlist['data'][modal][name], cv2.IMREAD_GRAYSCALE).astype(np.float16)
                # cv2.imshow('img', cv2.imread(self.datasetlist['data'][modal][name], cv2.IMREAD_GRAYSCALE))
                # cv2.waitKey(2000)
            else:
                name = list(self.datasetlist['label'])[index]
                label[:,:]=cv2.imread(self.datasetlist['label'][name], cv2.IMREAD_GRAYSCALE).astype(np.float16)

        #data augmentation
        if self.is_dataaug==True:
            data,label=self.augmentation(data,label)
        data = torch.from_numpy(np.ascontiguousarray(data))
        label = torch.from_numpy(np.ascontiguousarray(label))
        return data,label, name

    def __len__(self):
        return self.datanum
    # image augmentation
    def augmentation(self,image,annotation):
        # flipud
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = np.flipud(image[i,:,:])
            annotation= np.flipud(annotation)
        #fliplr
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = np.fliplr(image[i,:,:])
            annotation= np.fliplr(annotation)
            '''
        #noise
        if torch.randint(0, 4, (1,)) == 0:
            for i in range(self.modalitynum):
                image[i,:,:] = random_noise(image[i,:,:], mode='gaussian')
        '''
        return image,annotation
