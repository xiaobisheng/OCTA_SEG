import os
import cv2
import numpy as np
import open3d as o3d

root_path = '/Users/bishengwang/Desktop/WORK/medical_segmentation/Data/OCTA_6mm/Projection Maps'
def generate_depth_maps(data, i, threshold=10):
    data = data.transpose(2, 1, 0)

    # OCTA projection map
    img = np.average(data, axis=1) / 255
    cv2.imshow('img', img * 40)
    cv2.waitKey(200)

    # DCCM
    output_dccm = np.sum(data > threshold, axis=1)
    output_dccm = output_dccm / data.shape[1]
    cv2.imshow('DCCM', output_dccm)
    cv2.waitKey(2000)
    # save_path = root_path + '/OCTA_dccm_%d/%d.bmp' % (threshold, i)
    # cv2.imwrite(save_path, output_dccm)

    # DCIM
    data_ = data.transpose(0, 2, 1)
    indices = np.tile(np.arange(data_.shape[2]), (data_.shape[0], data_.shape[1], 1))  # 生成形状为 (W, H, C) 的通道索引矩阵
    valid_indices = np.where(data_ > threshold, indices, 0)  # 仅保留大于0的索引，其他填充 NaN
    output_dcim = np.sum(valid_indices, axis=2)
    num = np.sum(data_ > threshold, axis=2) + 0.01
    output_dcim = output_dcim / num
    output_dcim = output_dcim / data.shape[1]
    cv2.imshow('DCIM', output_dcim)
    cv2.waitKey(20000)
    # save_path = root_path + '/OCTA_dcim_%d/%d.bmp' % (threshold, i)
    # cv2.imwrite(save_path, output_dcim)

def visualize_3d(data):
    data = data[:, :, :] * 3
    W, H, C = data.shape
    threshold = 25  # 设定一个阈值，忽略低强度点
    coords = np.argwhere(data > threshold)  # 只取灰度值大于5的点
    colors = data[coords[:, 0], coords[:, 1], coords[:, 2]]
    colors = np.tile(colors[:, None], (1, 3)) / 255.0  # 归一化为RGB
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)  # 设置点坐标
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置颜色（灰度）
    # 可视化
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    for i in range(10001, 10300, 1):
        octa_data = np.load(root_path + '/OCTA_npy/%d.npy' % (i))
        visualize_3d(octa_data)
        generate_depth_maps(octa_data, i)