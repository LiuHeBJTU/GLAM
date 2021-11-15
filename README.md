# Joint Graph Learning and Matching for Semantic Feature Correspondence

## Overview of GLAM architecture
![image](https://user-images.githubusercontent.com/86004891/141059865-e6490804-6eae-4fdb-9499-ed4b36e5fa82.png)
For more details, please download our paper on [arXiv](https://arxiv.org/abs/2109.00240). If you have any question about the code, please contact me via the E-mail: liuhe1996@bjtu.edu.cn

## Requirements
pytorch           1.8.1+cu111  
torch_geometric   1.7.2

## Datasets
#### Pascal VOC Keypoint 
1. Download [VOC2011 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html) and make sure it looks like `data/PascalVOC/VOC2011` 
2. Download [keypoint annotation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz) for VOC2011 and make sure it looks like `data/PascalVOC/annotations`
3. The train/test split is available in `data/PascalVOC/voc2011_pairs.npz`
  
#### Willow Objects 
1. Download [Willow-ObjectClass](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip) dataset
2. Unzip the dataset and make sure it looks like `data/Willow`

#### SPair-71k
1. Download [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz) dataset
2. Unzip the dataset and make sure it looks like `data/SPair71k`

In implementation, the keypoint features are extracted by the standard VGG16-bn backbone networks in advance. The raw keypoint features for the three datasets are packed in the [KPT_features.zip](https://drive.google.com/file/d/14iApmo8u0XJ81-3OIz6Y-tVZAJaQokTT/view?usp=sharing). Unzip the package file and make sure it looks like `data/KPT_features`

## Experiments
### Training
To train the model on different datasets, please run  
`python train.py --dataset {Pascal_VOC, Willow, SPair_71k}  --lr 0.0001`
### Testing
The pretrained models are available on [[Google drive]](https://drive.google.com/file/d/1DsZl_VWmruVJ9W2cKaSP8ggtaRyHslOg/view?usp=sharing). 
To evaluate our model, please run  
`python eval.py --dataset {Pascal_VOC, Willow, SPair_71k}`

### Results

#### Willow
Car  |Duck  | Face |Mbike | Wbott. | AVG  | 
---- | ---- | ---- | ---- | ---- | -----|
99.0 | 99.0 |  100 |  100 |  100 |99.6 | 


#### Pascal VOC
aero | bike | bird | boat | bot. | bus  | car  | cat  | cha. | cow  | tab.|  dog | hor. | mbi. | per. | pla. | she. | sofa | tra. |  tv  | AVG  |
---- | ---- | ---- | ---- | ---- | -----| ---- | ---- | ---- | ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
72.3 | 76.8 | 84.3 | 77.4 | 94.9 | 95.7 | 93.8 | 85.9 | 72.6 | 87.9 | 100 | 86.2 | 85.2 | 85.3 | 71.4 | 98.9 | 83.8 | 80.5 | 98.8 | 92.8 | 86.2 |

##### Visualization of representative graph patterns learnt from Pascal VOC
![image](https://user-images.githubusercontent.com/86004891/141279508-a58c480b-ff21-4af7-ab86-8cda0d56062a.png)
![image](https://user-images.githubusercontent.com/86004891/141280054-648d71ab-5776-445e-aea1-daf9c509aa28.png)


#### SPair-71k
aero | bike | bird | boat | bott.| bus  |  car | cat  | chair| cow  | dog | hor. | mbi. | per. | plant| she. | train|  tv  | AVG  |
---- | ---- | ---- | ---- | ---- | -----| ---- | ---- | ---- | ---- | --- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
70.3 | 61.3 | 90.3 | 88.0 | 70.9 | 98.0 | 90.0 | 74.6 | 78.5 | 85.0 | 74.5| 76.9 | 75.8 | 79.6 | 99.2 | 79.1 | 92.2 | 99.9 | 82.5 |


## BibTeX
If you use this code for your research, please consider citing:

@article{liu2021joint,  
  title={Joint graph learning and matching for semantic feature correspondence},  
  author={Liu, He and Wang, Tao and Li, Yidong and Lang, Congyan and Jin, Yi and Ling, Haibin},  
  journal={arXiv preprint arXiv:2109.00240},  
  year={2021}  
} 
