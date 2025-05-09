import numpy as np


def rotate_matrix(theta):
    m = np.zeros((2, 2))  # 创建一个2x2的零矩阵
    m[0, 0] = np.cos(theta)  # 第一行第一列为cos(θ)
    m[0, 1] = -np.sin(theta)  # 第一行第二列为-sin(θ)
    m[1, 0] = np.sin(theta)  # 第二行第一列为sin(θ)
    m[1, 1] = np.cos(theta)  # 第二行第二列为cos(θ)
    return m



def Rotate_DA(x, y):
    [N, L, C] = np.shape(x)  # 获取输入数据x的形状，假设是(N, L, C)的形状
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi / 2))  # 将x旋转90度
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))  # 将x旋转180度
    x_rotate3 = np.matmul(x, rotate_matrix(3 * np.pi / 2))  # 将x旋转270度

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))  # 将原始数据和旋转后的数据堆叠成一个新的数据集

    y_DA = np.tile(y, (1, 4))  # 将标签y复制4次，适应数据量的增加
    y_DA = y_DA.T  # 转置，匹配数据的维度
    y_DA = y_DA.reshape(-1)  # 将标签展平为一个一维数组
    y_DA = y_DA.T  # 将标签数据还原
    return x_DA, y_DA  # 返回旋转后的数据和标签


# 这段代码实现了对输入数据的旋转增强，
# 将数据旋转90度、180度、270度后堆叠起来，生成一个更大的数据集。
# 同时，它扩展了标签，以匹配旋转后的数据集大小。
# 这是一种常见的数据增强技术，特别适用于图像和其他二维数据。