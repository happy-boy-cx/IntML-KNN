import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop

from Openset_RFFI_TIFS.augment import Rotate_DA
from Openset_RFFI_TIFS.get_dataset import load_train_dataset_k_shot
from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from deep_learning_models import TripletNet, identity_loss

def rand_bbox(size, lamb):
    length = size[2]
    cut_rate = 1.-lamb
    cut_length = int(length*cut_rate)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2

def train_feature_extractor(
        file_path='./dataset/Train/dataset_training_aug.h5',
        dev_range=np.arange(0, 1, dtype=int),
        pkt_range=np.arange(0, 32, dtype=int),
        snr_range=np.arange(20, 80)
):
    LoadDatasetObj = LoadDataset()

    # Load preamble IQ samples and labels.
    x_train, label = LoadDatasetObj.load_iq_samples(file_path,
                                                 dev_range,
                                                 pkt_range)

    # 获取实部和虚部
    real_part = np.real(x_train)
    imag_part = np.imag(x_train)

    # 将实部和虚部沿新维度堆叠
    x_train = np.stack((real_part, imag_part), axis=-1)

    x_train, label = Rotate_DA(x_train, label)

    min_value = x_train.min()

    max_value = x_train.max()

    x_train = (x_train - min_value) / (max_value - min_value)

    datax = x_train

    # new edition: Triplet + Rotate + CutMix
    lam = np.random.beta(1, 1)  # beta distribution to generate cropped area
    print(type(datax))  # 应该是 <class 'torch.Tensor'>
    index = np.random.permutation(datax.shape[0])  # datax.shape[0] 获取样本数量
    bbx1, bbx2 = rand_bbox(datax.shape, lam)
    datax[:, :, bbx1:bbx2] = datax[index, :, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) / datax.shape[-1])

    data = datax[:, :, 0] + 1j * datax[:, :, 1]

    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    # Convert time-domain IQ samples to channel-independent spectrograms.    将时域IQ样本转换为信道无关谱图。
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)

    # Specify hyper-parameters during training.
    margin = 0.1  # 三元组损失的边际值。
    batch_size = 32  # 每个批次的样本数量。
    patience = 20  # 早停时容忍的轮数。

    TripletNetObj = TripletNet()  # 实例化三元组网络类。
    data = data[0:1, :, :]  # 保持 (1, 1000, 1000)
    # Create an RFF extractor.
    feature_extractor = TripletNetObj.feature_extractor(data.shape)

    # Create the Triplet net using the RFF extractor.
    triplet_net = TripletNetObj.create_triplet_net(feature_extractor, margin)

    count=triplet_net.summary()
    print(count)

    # Create callbacks during training. The training stops when validation loss
    # does not decrease for 30 epochs.
    early_stop = EarlyStopping('val_loss',
                               min_delta=0,
                               patience=
                               patience)

    reduce_lr = ReduceLROnPlateau('val_loss',
                                  min_delta=0,
                                  factor=0.2,
                                  patience=10,
                                  verbose=1)
    callbacks = [early_stop, reduce_lr]

    # Split the dasetset into validation and training sets.     将数据和标签划分为训练集和验证集，测试集占10%。
    data_train, data_valid, label_train, label_valid = train_test_split(data,
                                                                        label,
                                                                        test_size=0.1,
                                                                        shuffle=True)
    del data, label

    # Create the trainining generator.
    train_generator = TripletNetObj.create_generator(batch_size,
                                                     dev_range,
                                                     data_train,
                                                     label_train)
    # Create the validation generator.
    valid_generator = TripletNetObj.create_generator(batch_size,
                                                     dev_range,
                                                     data_valid,
                                                     label_valid)

    # Use the RMSprop optimizer for training.
    opt = RMSprop(learning_rate=1e-3)
    triplet_net.compile(loss=identity_loss, optimizer=opt)

    # Start training.
    history = triplet_net.fit(train_generator,
                              steps_per_epoch=data_train.shape[0] // batch_size,
                              epochs=1000,
                              validation_data=valid_generator,
                              validation_steps=data_valid.shape[0] // batch_size,
                              verbose=1,
                              callbacks=callbacks)

    return feature_extractor

train_feature_extractor()