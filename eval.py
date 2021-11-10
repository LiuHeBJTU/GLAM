import torch
import numpy as np

from torch.utils.data import DataLoader

from GLAM_model import Model
from utils.collate import collate_fn

from dataset.Pascal_voc import PascalVOC
from dataset.Willow import Willow
from dataset.Spair import Spair
from evaluate.evaluate_Willow import willow_test
from evaluate.evaluate_PascalVOC import pascal_test
from evaluate.evaluate_Spair import spair_test

torch.backends.cudnn.benchmark = True

dataset = "Pascal_VOC"  # Pascal_VOC, Willow, SPair_71k
NUM_HEAD = 8
Dq = Dk = Dv = Dinner = 1024
acc_all = []

if dataset == "Pascal_VOC":
    dataset_train = PascalVOC("train")
    dataset_test = PascalVOC("test")
    model_file = "pretrained_models/Pascal_VOC_params.pth"
elif dataset == "Willow":
    dataset_train = Willow("train")
    dataset_test = Willow("test")
    model_file = "pretrained_models/Willow_params.pth"
else:
    dataset_train = Spair("train")
    dataset_test = Spair("test")
    model_file = "pretrained_models/SPair_71k_params.pth"

dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

GLAM = Model(d_model=Dq, d_inner=Dinner, n_head=NUM_HEAD, d_k=Dk, d_v=Dv)
pretrained_params = torch.load(model_file)
GLAM.load_state_dict(pretrained_params)
GLAM.eval()

if torch.cuda.is_available():
    GLAM = GLAM.cuda()

with torch.no_grad():
    if dataset == "Pascal_VOC":
        acc_dict = pascal_test(model=GLAM, dataset_test=dataloader_test)
    elif dataset == "Willow":
        acc_dict = willow_test(model=GLAM, dataset_test=dataloader_test)
    else:
        acc_dict = spair_test(model=GLAM, dataset_test=dataloader_test)

    for key, value in acc_dict.items():
        acc = np.mean(value)
        print(key + " " + str(np.round(acc, 4)))
        acc_all.append(acc)
    print("AVG " + str(np.round(np.mean(acc_all), 4)))
