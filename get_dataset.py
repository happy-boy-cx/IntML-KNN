from Openset_RFFI_TIFS.dataset_preparation import LoadDataset, ChannelIndSpectrogram
import matplotlib.pyplot as plt
import numpy as np
import random

def load_test_dataset(
        file_path = './dataset/Train/dataset_training_aug.h5',
        dev_range = np.arange(0,30, dtype = int),
        pkt_range = np.arange(50,100, dtype = int)
                            ):
    LoadDatasetObj = LoadDataset()

    # Load preamble IQ samples and labels.
    data, label = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range,
                                                 pkt_range)
    label = label.astype(np.uint8)  # 将标签数据转换为无符号8位整数类型
    # 获取实部和虚部
    real_part = np.real(data)
    imag_part = np.imag(data)

    # 将实部和虚部沿新维度堆叠
    x_train_k_shot_expanded = np.stack((real_part, imag_part), axis=-1)

    print(x_train_k_shot_expanded.shape)  # 输出 (300, 8192, 2)

    return x_train_k_shot_expanded, label


def load_val_dataset(
        file_path = './dataset/Train/dataset_training_aug.h5',
        dev_range = np.arange(0,30, dtype = int),
        pkt_range = np.arange(48,50, dtype = int)
                            ):
    LoadDatasetObj = LoadDataset()

    # Load preamble IQ samples and labels.
    data, label = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range,
                                                 pkt_range)
    label = label.astype(np.uint8)  # 将标签数据转换为无符号8位整数类型

    # 获取实部和虚部
    real_part = np.real(data)
    imag_part = np.imag(data)

    # 将实部和虚部沿新维度堆叠
    x_train_k_shot_expanded = np.stack((real_part, imag_part), axis=-1)

    print(x_train_k_shot_expanded.shape)  # 输出 (300, 8192, 2)

    return x_train_k_shot_expanded,label

def load_train_dataset_k_shot(
        file_path = './dataset/Train/dataset_training_aug.h5',
        dev_range = np.arange(0,10, dtype = int),
        pkt_range = np.arange(0,1000, dtype = int)
                            ):
    num=10
    k_shot=50
    LoadDatasetObj = LoadDataset()

    # Load preamble IQ samples and labels.
    data, label = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range,
                                                 pkt_range)
    label = label.astype(np.uint8)  # 将标签数据转换为无符号8位整数类型

    random_index_shot = []  # 初始化一个空列表，用于存储k-shot采样的索引
    for i in range(num):  # 遍历每个类别
        index_shot = [index for index, value in enumerate(label) if value == i]  # 获取所有标签值为i的索引
        random_index_shot += random.sample(index_shot, k_shot)  # 从每个类别中随机选择k_shot个样本的索引，并加入到random_index_shot列表
    random.shuffle(random_index_shot)  # 随机打乱所有采样的索引
    x_train_k_shot = data[random_index_shot, :]  # 使用这些索引来选择训练集中的样本
    y_train_k_shot = label[random_index_shot]  # 使用这些索引来选择训练集中的标签
    # 获取实部和虚部
    real_part = np.real(x_train_k_shot)
    imag_part = np.imag(x_train_k_shot)

    # 将实部和虚部沿新维度堆叠
    x_train_k_shot_expanded = np.stack((real_part, imag_part), axis=-1)

    print(x_train_k_shot_expanded.shape)  # 输出 (300, 8192, 2)


    return x_train_k_shot_expanded,y_train_k_shot


load_test_dataset()