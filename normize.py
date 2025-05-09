import numpy as np
import matplotlib.pyplot as plt

from Openset_RFFI_TIFS.get_dataset import load_train_dataset_k_shot


def train_feature_extractor():
    # 加载训练数据和标签
    x_train, label = load_train_dataset_k_shot()

    # 添加高斯噪声到IQ样本
    x_train = x_train[:, :, 0] + 1j * x_train[:, :, 1]

    # 获取实部和虚部
    real_part = np.real(x_train)
    imag_part = np.imag(x_train)

    # 将实部和虚部堆叠
    x_train = np.stack((real_part, imag_part), axis=-1)

    # 归一化
    min_value = x_train.min()
    max_value = x_train.max()
    x_train = (x_train - min_value) / (max_value - min_value)

    # 绘制归一化后的图像
    plot_normalized_image(x_train)

    return x_train, label


import matplotlib.pyplot as plt
import numpy as np


def plot_normalized_image(x_train):
    # 假设x_train的形状是 (样本数, 时间序列长度, 2)，其中2是实部和虚部
    sample_index = 0  # 可以修改为其他样本索引
    sample = x_train[sample_index]

    # 提取实部
    real_part = sample[:, 0]

    # 创建图形并绘制实部
    plt.figure(figsize=(10, 6))
    fs = 1000  # 假设采样率为1000 Hz
    time = np.arange(len(real_part)) / fs  # 时间序列（单位：秒）

    # 只绘制实部
    plt.plot(time, real_part, color='#0072bd', linestyle='-', linewidth=1)

    # 添加图表标题和标签
    plt.ylabel('Amplitude')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


# 你可以调用train_feature_extractor函数来训练并绘制归一化后的图像
x_train, label = train_feature_extractor()
