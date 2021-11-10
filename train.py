import os
import torch
import argparse
import numpy as np

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from GLAM_model import Model
from utils.losses import loss_
from utils.collate import collate_fn
from utils.compute_accuracy import accuracy
from dataset.Pascal_voc import PascalVOC
from dataset.Willow import Willow
from dataset.Spair import Spair
from evaluate.evaluate_Willow import willow_test
from evaluate.evaluate_PascalVOC import pascal_test
from evaluate.evaluate_Spair import spair_test

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Joint Graph Learning and Matching for Feature Correspondence')
    parser.add_argument('--dataset', type=str, default='Willow')  # Pascal_VOC, Willow, SPair_71k
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    dataset = args.dataset
    NUM_HEAD = 8
    cur_lr = args.lr
    best_acc = 0
    Dq = Dk = Dv = Dinner = 1024
    loss_train, acc_train = [], []

    if dataset == "Pascal_VOC":
        dataset_train = PascalVOC("train")
        dataset_test = PascalVOC("test")
        BATCH_SIZE = 128

    elif dataset == "Willow":
        dataset_train = Willow("train")
        dataset_test = Willow("test")
        BATCH_SIZE = 16

    else:
        dataset_train = Spair("train")
        dataset_test = Spair("test")
        BATCH_SIZE = 128

    save_path = "save/" + dataset + "/" + "_BS_" + str(BATCH_SIZE) + "_RL_" + str(cur_lr)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    GLAM = Model(d_model=Dq, d_inner=Dinner, n_head=NUM_HEAD, d_k=Dk, d_v=Dv)

    if torch.cuda.is_available():
        GLAM = GLAM.cuda()

    # optimizer=Adam(params=TFer_Net.parameters(),lr=cur_lr,betas=(0.5, 0.999))
    optimizer = SGD(params=GLAM.parameters(), lr=cur_lr, momentum=0.9)

    for epoch in range(30):
        GLAM.train()
        for idx, data in enumerate(dataloader_train):
            optimizer.zero_grad()
            descs0, descs1, pts0, pts1, X, atten_mask, key_mask, num_node_list = data

            match_matrix = GLAM(descs0, descs1, pts0, pts1, atten_mask, key_mask)

            loss_tr, perm_predicted = loss_(outputs=match_matrix, targets=X, num_nodes_list=num_node_list)
            loss_tr.backward()
            optimizer.step()

            loss_train.append(loss_tr.data.cpu().numpy().item())
            acc_train.append(accuracy(perm_predicted.data.cpu().numpy(), X.cpu().numpy(), num_node_list))

            if (idx + 1) % 100 == 0:
                print(dataset + " Epoch {:05d}, # {:05d}, loss_train {:.4f}, acc_train {:.4f},  LR {:.10f}".format(
                    epoch + 1, idx + 1, np.round(np.mean(loss_train), 4), np.round(np.mean(acc_train), 4), cur_lr))
                loss_train.clear()
                acc_train.clear()

        GLAM.eval()
        with torch.no_grad():
            if dataset == "Pascal_VOC":
                acc_dict = pascal_test(model=GLAM, dataset_test=dataloader_test)
            elif dataset == "Willow":
                acc_dict = willow_test(model=GLAM, dataset_test=dataloader_test)
            else:
                acc_dict = spair_test(model=GLAM, dataset_test=dataloader_test)
            acc_all = []
            for key, value in acc_dict.items():
                acc_all.append(np.mean(value))
            acc_test = np.mean(acc_all)
            print(dataset + " Epoch {:05d}, acc_test {:.4f}, LR {:.10f}".format(
                    epoch + 1, np.round(acc_test, 4), cur_lr))

            if acc_test > best_acc:
                best_acc = acc_test
                torch.save(GLAM.state_dict(), save_path + "/" + dataset + "_params.pth")

            loss_train.clear()
            acc_train.clear()

        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.99, 0.000001)
            cur_lr = param_group['lr']
        GLAM.train()