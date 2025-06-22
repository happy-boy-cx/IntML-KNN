# IntML-KNN: A Few-Shot Radio Frequency Fingerprint Identification Scheme for Lora Devices

# Paper
The code corresponds to the paper [https://ieeexplore.ieee.org/document/10999054](https://ieeexplore.ieee.org/document/10999054)

X. Cao, W. Tan, Q. Gao, Z. Hu and C. Li, "IntML-KNN: A Few-Shot Radio Frequency Fingerprint Identification Scheme for LoRa Devices," in IEEE Signal Processing Letters, vol. 32, pp. 2259-2263, 2025, doi: 10.1109/LSP.2025.3569211. 

# Requirement
python == 3.6.13

h5py == 3.1.0

keras == 2.6.0

matplotlib == 3.3.4

numpy == 1.19.2

scikit_learn == 1.6.1

scipy == 1.5.2

seaborn == 0.11.1

tensorflow == 2.6.2

torch == 1.10.2

# Abstract
Deep learning (DL) is widely used in radio frequency fingerprint identification (RFFI). However, in few-shot case, traditional DL-based RFFI need to construct auxiliary dataset to realize radio frequency fingerprint identification. To address this issue, we propose a few-shot RFFI (FS-RFFI) method based on interpolation metric learning and KNN (IntML-KNN). Specifically, the method first extends the dataset with data augmentation, and CutMix interpolation. Secondly, combining with metric learning to enhance the generalization capacity of the model. Finally, KNN algorithm is designed to realize device classification and detection. The proposed IntML-KNN method is verified on the commercial available LoRa dataset. The experimental results indicate that the proposed scheme exhibits strong classification and generalization performance in FS-RFFI. Meanwhile, the identification rate of the proposed IntML-KNN reaches 97.00% with only 10% samples.

# Framework of IntML-KNN
![Alt](https://i-blog.csdnimg.cn/direct/cd30c9f3332f4d309299130647965bc9.png#pic_center)
# Dataset and Experimental Setup

The model in IntML-KNN is based on PyTorch, and the simulation platform is GTX1080Ti. We use real LoRa datasets to verify our model, and the detailed signal acquisition process can be found in paper [15]. We used 25 of these LoRa signal datasets for simulation. Among them, the first 10 are used for device classification experiments, and the last 15 are used for rogue device detection experiments. In the process of model training, we carried out simulation experiments with small samples according to 5%, 10% and 15% of the total data respectively. Training data, test data, and validation data are distributed in an 8:1:1 ratio.

[15] G. Shen, J. Zhang, A. Marshall and J. R. Cavallaro, "Towards scalable and channel-robust radio frequency fingerprint identification for LoRa," IEEE Trans. Inf. Forensics Security, vol. 17, pp. 774-787, 2022.
# Classification Accuracy


Method     | 5%     | 10%     | 15%
-------- | ----- | ----- | -----
CVCNN | 16.80 | 42.20 | 56.00
CVCNN-AT | 42.20 | 61.90 | 74.50
MAT-CL  | 53.40 | 72.80 | 89.20
IntML-KNN  | 83.90 | 97.00 | 98.50

# Complexity of Model

Method     | Params     | MACs     | Storage(M)     | Iter time/s
-------- | ----- | ----- | ----- | -----
IntML-KNN |$2.11\times10^6$ | $2.76\times10^8$ | 6.84 | 0.71

# Detection of Rogue Devices

![Alt](https://i-blog.csdnimg.cn/direct/24e37db07d3b4f25824bfe11f455afbd.png#pic_center)
# Generalization Ability for Classification

![Alt](https://i-blog.csdnimg.cn/direct/33c7459ad1d5484cad615267d0ac769e.png#pic_center)
# Email

If you have any question, please feel free to contact us by e-mail (<gs.xcao23@gzu.edu.cn>).
