# Joint Graph Learning and Matching for Semantic Feature Correspondence

## Overview of GLAM architecture
![image](https://user-images.githubusercontent.com/86004891/141059865-e6490804-6eae-4fdb-9499-ed4b36e5fa82.png)
For more details, please download our paper on [arXiv](https://arxiv.org/abs/2109.00240)

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
The pretrained models are available on [[Google drive]](https://drive.google.com/file/d/1ndqEblJAPTfaJyPOU3mzn0BP51r0ouW4/view?usp=sharing). 
To evaluate our model, please run  
`python eval.py --dataset {Pascal_VOC, Willow, SPair_71k}`

## BibTeX
If you use this code for your research, please consider citing:

@article{liu2021joint,  
  title={Joint graph learning and matching for semantic feature correspondence},  
  author={Liu, He and Wang, Tao and Li, Yidong and Lang, Congyan and Jin, Yi and Ling, Haibin},  
  journal={arXiv preprint arXiv:2109.00240},  
  year={2021}  
} 
