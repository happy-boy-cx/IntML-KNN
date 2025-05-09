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
from tensorflow.keras.optimizers import RMSprop

from Openset_RFFI_TIFS.augment import Rotate_DA
from Openset_RFFI_TIFS.get_dataset import load_train_dataset_k_shot
from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from deep_learning_models import TripletNet, identity_loss


#%%

def rand_bbox(size, lamb):
    length = size[2]
    cut_rate = 1.-lamb
    cut_length = int(length*cut_rate)
    cx = np.random.randint(length)
    bbx1 = np.clip(cx - cut_length//2, 0, length)
    bbx2 = np.clip(cx + cut_length//2, 0, length)
    return bbx1, bbx2

def train_feature_extractor(
        dev_range = np.arange(0,10, dtype = int)):

    x_train, label = load_train_dataset_k_shot()
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
    margin = 0.1    #三元组损失的边际值。
    batch_size = 32 #每个批次的样本数量。
    patience = 20   #早停时容忍的轮数。
    
    TripletNetObj = TripletNet()    #实例化三元组网络类。
    
    # Create an RFF extractor.
    feature_extractor = TripletNetObj.feature_extractor(data.shape)
    
    # Create the Triplet net using the RFF extractor.
    triplet_net = TripletNetObj.create_triplet_net(feature_extractor, margin)

    # Create callbacks during training. The training stops when validation loss 
    # does not decrease for 30 epochs.
    early_stop = EarlyStopping('val_loss', 
                               min_delta = 0, 
                               patience = 
                               patience)
    
    reduce_lr = ReduceLROnPlateau('val_loss', 
                                  min_delta = 0, 
                                  factor = 0.2, 
                                  patience = 10, 
                                  verbose=1)
    callbacks = [early_stop, reduce_lr]
    
    # Split the dasetset into validation and training sets.     将数据和标签划分为训练集和验证集，测试集占10%。
    data_train, data_valid, label_train, label_valid = train_test_split(data, 
                                                                        label, 
                                                                        test_size=0.1, 
                                                                        shuffle= True)
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
    triplet_net.compile(loss = identity_loss, optimizer = opt)
    inference_times = []
    # Start training.
    history = triplet_net.fit(train_generator,
                              steps_per_epoch = data_train.shape[0]//batch_size,
                              epochs = 1000,
                              validation_data = valid_generator,
                              validation_steps = data_valid.shape[0]//batch_size,
                              verbose=1, 
                              callbacks = callbacks)


    return feature_extractor

def test_classification(
        file_path_enrol,
        file_path_clf,
        feature_extractor_name,
        dev_range_enrol = np.arange(0,10, dtype = int),
        pkt_range_enrol = np.arange(0,200, dtype = int),
        dev_range_clf = np.arange(0,10, dtype = int),
        pkt_range_clf = np.arange(200,300, dtype = int)
                        ):
    
    # Load the saved RFF extractor.
    feature_extractor = load_model(feature_extractor_name, compile=False)

    
    LoadDatasetObj = LoadDataset()
    
    # Load the enrollment dataset. (IQ samples and labels)
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol, 
                                                             dev_range_enrol, 
                                                             pkt_range_enrol)
    
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    
    # Convert IQ samples to channel independent spectrograms. (enrollment data)
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)
    
    # # Visualize channel independent spectrogram
    # plt.figure()
    # sns.heatmap(data_enrol[0,:,:,0],xticklabels=[], yticklabels=[], cmap='Blues', cbar=False)
    # plt.gca().invert_yaxis()
    # plt.savefig('channel_ind_spectrogram.pdf')
    
    # Extract RFFs from channel independent spectrograms.
    feature_enrol = feature_extractor.predict(data_enrol)
    del data_enrol
    
    # Create a K-NN classifier using the RFFs extracted from the enrollment dataset.
    knnclf=KNeighborsClassifier(n_neighbors=15,metric='euclidean')
    knnclf.fit(feature_enrol, np.ravel(label_enrol))
    
    
    # Load the classification dataset. (IQ samples and labels)
    data_clf, true_label = LoadDatasetObj.load_iq_samples(file_path_clf, 
                                                         dev_range_clf, 
                                                         pkt_range_clf)
    
    # Convert IQ samples to channel independent spectrograms. (classification data)
    data_clf = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_clf)

    # Extract RFFs from channel independent spectrograms.
    feature_clf = feature_extractor.predict(data_clf)
    del data_clf
    
    # Make prediction using the K-NN classifier.
    pred_label = knnclf.predict(feature_clf)

    # Calculate classification accuracy.
    acc = accuracy_score(true_label, pred_label)
    print('Overall accuracy = %.4f' % acc)
    
    return pred_label, true_label, acc




def test_rogue_device_detection(
    feature_extractor_name,
    file_path_enrol = './dataset/Test/dataset_residential.h5',
    dev_range_enrol = np.arange(30,40, dtype = int),
    pkt_range_enrol = np.arange(0,100, dtype = int),
    file_path_legitimate = './dataset/Test/dataset_residential.h5',
    dev_range_legitimate = np.arange(30,40, dtype = int),
    pkt_range_legitimate = np.arange(100,200, dtype = int),
    file_path_rogue = './dataset/Test/dataset_rogue.h5',
    dev_range_rogue = np.arange(40,45, dtype = int),
    pkt_range_rogue = np.arange(0,100, dtype = int),
    ):

    
    def _compute_eer(fpr,tpr,thresholds):
        '''
        _COMPUTE_EER returns equal error rate (EER) and the threshold to reach
        EER point.
        '''
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        return eer, thresholds[min_index]
    
    # Load RFF extractor.
    feature_extractor = load_model(feature_extractor_name, compile=False)
    
    LoadDatasetObj = LoadDataset()
    
    # Load enrollment dataset.
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol, 
                                                             dev_range_enrol, 
                                                             pkt_range_enrol)
    
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    
    # Convert IQ samples to channel independent spectrograms.
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)
    
    # Extract RFFs from cahnnel independent spectrograms.
    feature_enrol = feature_extractor.predict(data_enrol)
    del data_enrol
    
    # Build a K-NN classifier.
    knnclf=KNeighborsClassifier(n_neighbors=15,metric='euclidean')
    knnclf.fit(feature_enrol, np.ravel(label_enrol))
    
    # Load the test dataset of legitimate devices.
    data_legitimate, label_legitimate = LoadDatasetObj.load_iq_samples(file_path_legitimate, 
                                                                       dev_range_legitimate, 
                                                                       pkt_range_legitimate)
    # Load the test dataset of rogue devices.
    data_rogue, label_rogue = LoadDatasetObj.load_iq_samples(file_path_rogue, 
                                                             dev_range_rogue, 
                                                             pkt_range_rogue)
    
    # Combine the above two datasets into one dataset containing both rogue
    # and legitimate devices.
    data_test = np.concatenate([data_legitimate,data_rogue])
    label_test = np.concatenate([label_legitimate,label_rogue])
    
    # Convert IQ samples to channel independent spectrograms.
    data_test = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)

    # Extract RFFs from channel independent spectrograms.
    feature_test = feature_extractor.predict(data_test)
    del data_test

    # Find the nearest 15 neighbors in the RFF database and calculate the 
    # distances to them.
    distances, indexes = knnclf.kneighbors(feature_test)
    
    # Calculate the average distance to the nearest 15 neighbors.
    detection_score = distances.mean(axis =1)

    # Label the packets sent from legitimate devices as 1. The rest are sent by rogue devices
    # and are labeled as 0.
    true_label = np.zeros([len(label_test),1])
    true_label[(label_test <= dev_range_legitimate[-1]) & (label_test >= dev_range_legitimate[0])] = 1
    
    # Compute receiver operating characteristic (ROC).
    fpr, tpr, thresholds = roc_curve(true_label, detection_score, pos_label = 1)
    
    # The Euc. distance is used as the detection score. The lower the value, 
    # the more similar it is. This is opposite with the probability or confidence 
    # value used in scikit-learn roc_curve function. Therefore, we need to subtract 
    # them from 1.
    fpr = 1-fpr  
    tpr = 1-tpr

    # Compute EER.
    eer, _ = _compute_eer(fpr,tpr,thresholds)
    
    # Compute AUC.
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, eer

if __name__ == '__main__':
    
    # Specifies what task the program runs for. 
    # 'Train'/'Classification'/'Rogue Device Detection/Predict Device'
    run_for = 'Rogue Device Detection'
    
    if run_for == 'Train':

        # Train an RFF extractor.
        feature_extractor = train_feature_extractor()
        # Save the trained model.
        feature_extractor.save('Extractor_small_15%-2.h5')
        
    
    elif run_for == 'Classification':
        
        # Specify the device index range for classification.
        test_dev_range = np.arange(0,10, dtype = int)
        
        # Perform the classification task.
        pred_label, true_label, acc = test_classification(file_path_enrol = 
                                                          './dataset/Test/dataset_seen_devices.h5',
                                                          file_path_clf =
                                                          './dataset/Test/dataset_seen_devices.h5',
                                                          feature_extractor_name =
                                                          './models/Extractor_small_15%.h5')
        
        # Plot the confusion matrix.
        conf_mat = confusion_matrix(true_label, pred_label)
        classes = test_dev_range + 1
        
        plt.figure()
        sns.heatmap(conf_mat, annot=True, 
                    fmt = 'd', cmap='Blues',
                    cbar = False,
                    xticklabels=classes, 
                    yticklabels=classes)
        plt.xlabel('Predicted label', fontsize = 20)
        plt.ylabel('True label', fontsize = 20)
        plt.show()
        
    elif run_for == 'Rogue Device Detection':

        # Perform rogue device detection task using three RFF extractors.
        fpr, tpr, roc_auc, eer = test_rogue_device_detection('./models/Extractor_small_10%.h5')
        fpr4, tpr4, roc_auc4, eer4 = test_rogue_device_detection('./models/Extractor_small_MAT_CL.h5')
        fpr2, tpr2, roc_auc2, eer2 = test_rogue_device_detection('./models/Extractor_small_CVCNN_AT.h5')
        fpr3, tpr3, roc_auc3, eer3 = test_rogue_device_detection('./models/Extractor_small_CVCNN.h5')

        # Plot the ROC curves.
        plt.figure(figsize=(4.8, 2.8))
        plt.xlim(-0.01, 1.02)
        plt.ylim(-0.01, 1.02)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='IntML-KNN, ERR = ' +
                                 str(round(roc_auc, 3)) + ', AUC = ' + str(round(roc_auc, 3)), color='r')
        plt.plot(fpr4, tpr4, label='MAT-CL, ERR = ' +
                                   str(round(roc_auc4, 3)) + ', AUC = ' + str(round(eer4, 3)), color='y')
        plt.plot(fpr2, tpr2, label='CVCNN-AT, ERR = ' +
                                   str(round(roc_auc2, 3)) + ', AUC = ' + str(round(eer2, 3)), color='g')
        plt.plot(fpr3, tpr3, label='CVCNN, ERR = ' +
                                   str(round(roc_auc3, 3)) + ', AUC = ' + str(round(eer3, 3)), color='b')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc=4)
        # plt.savefig('roc_curve.pdf',bbox_inches='tight')
        plt.show()










