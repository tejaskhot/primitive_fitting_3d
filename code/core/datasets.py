import numpy as np
import random
import torch
import torch.utils.data as data
from plyfile import PlyData
import pc_utils
import glob

def random_jitter(pcd, sigma=0.02, clip=0.05):
    return pcd + np.clip(sigma * np.random.normal(size=pcd.shape), -clip, clip)

def Rotate2D(pts,cnt,ang=np.pi/4):
    '''pts = {} Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts-cnt, np.array([[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]]))+cnt

class BuildingDataloader(data.Dataset):
    def __init__(self, data_path, params_path, dtype, num_data=None, num_primitives=3, num_params=6, num_channels=3, batch_size=32):
        # this is a list of names of all pc files
        self.num_primitives = num_primitives
        self.num_channels = num_channels
        self.num_params = num_params
        self.data = []
        for fname in glob.iglob(data_path+dtype+'/*.ply'):
            k = fname[len(data_path)+len(dtype)+1:-4]
            count = int(k.split('_')[1])
            if count > num_primitives: continue
            self.data.append((k,fname))
        self.data = self.data[:(len(self.data)//batch_size)*batch_size]
        # self.params is a dict with each key being the name of the pc file
        self.params = np.load(params_path).item()
        if num_data:
            self.data = self.data[:num_data]

    def __getitem__(self, index):
        k = self.data[index][0]
        fname = self.data[index][1]
        points = PlyData.read(fname)
        points = np.asarray(points['vertex'].data.tolist(), dtype=np.float32)
        idx = np.random.randint(points.shape[0], size=4096)
        points = points[idx,:self.num_channels]
        # add noise to the point cloud
        # points = random_jitter(points)
        points = points.astype(np.float32)

        center = (points.max(0)+points.min(0))/2
        points = points - center
        scale = np.max(points.max(0)-points.min(0))/2
        points = points/scale

        # points = np.random.choice(points,(4096,3),replace=False)
        params = self.params[k].astype(np.float32)
        params[:,:3] -= center
        params[:,:6] /=scale

        probs = len(params)-1
        count = len(params)
        # add zero padding
        pad_size = self.num_primitives - len(params)
        params = np.pad(params,((0,pad_size),(0,0)),'constant')

        if self.num_params<7:
            # get rid of rotations
            points[:,:2] = Rotate2D(points[:,:2], [0,0], -params[0][6])
            params[:,:2] = Rotate2D(params[:,:2], [0,0], -params[0][6])
            params = params[:,:self.num_params]

        return points, params, probs, count

    def __len__(self):
        return len(self.data)

class BuildingGroupLoader(data.Dataset):
    def __init__(self, data_path, params_path, labels_path, dtype, num_data=None, num_channels=3):
        # this is a list of names of all pc files
        self.data = []
        self.num_channels = num_channels
        for fname in glob.iglob(data_path+dtype+'/*.ply'):
            k = fname[len(data_path)+len(dtype)+1:-4]
            self.data.append((k,fname))
        self.labels = {}
        for fname in glob.iglob(labels_path+'/*.npy'):
            k = fname[len(labels_path):-4]
            self.labels[k] = fname
        if num_data:
            self.data = self.data[:num_data]

    def __getitem__(self, index):
        k = self.data[index][0]
        fname = self.data[index][1]
        points = PlyData.read(fname)
        points = np.asarray(points['vertex'].data.tolist(), dtype=np.float32)[:,:self.num_channels]

        # add noise to the point cloud
        points = random_jitter(points)
        points = points.astype(np.float32)

        # group labels
        group_labels = np.load(self.labels[k])
        pts_group_label, pts_group_mask, group_counts = pc_utils.convert_groupandcate_to_one_hot(group_labels, no_batch=True, num_groups=4)

        return points, pts_group_label, group_counts

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    data_path = '../data/lod/pcd_ply/'
    params_path = '../data/lod/params.npy'
    dataloader = BuildingDataloader(data_path, params_path, num_primitives=3, dtype='train')
    # labels_path = '../data/instances/labels/'
    # dataloader = BuildingGroupLoader(data_path, params_path, labels_path, 'train')
    for i in range(10):
        points, params, probs, count = dataloader[0]
        # points, pts_group_label, group_counts = dataloader[i]
        print('len(dataloader) : {}'.format(len(dataloader)))
        print('points.shape : {}'.format(points.shape))
        print('params.shape : {}'.format(params.shape))
        # print('pts_group_label.shape : {}'.format(pts_group_label.shape))
        # print('group_counts.shape : {}'.format(group_counts.shape))
        print('='*50)
    # print('probs.shape : {}'.format(probs.shape))
    for i in range(10):
        points, params, probs, count = dataloader[i]
        print('len(dataloader) : {}'.format(len(dataloader)))
        print('points.shape : {}'.format(points.shape))
        print('params.shape : {}'.format(params.shape))
