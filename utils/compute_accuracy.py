import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

def accuracy(perm_pred,perm_gt,num_node_list):
    correct_avg=[]
    batchsize=perm_gt.shape[0]
    for idx in range(batchsize):
        num_nodes=num_node_list[idx]
        perm_g= perm_gt[idx][:num_nodes,:num_nodes]
        perm_p= perm_pred[idx][:num_nodes,:num_nodes]

        result= np.zeros((num_nodes,num_nodes))
        row_ind, col_ind = linear_sum_assignment(-perm_p)
        result[row_ind, col_ind]=1

        correct=np.sum((result==perm_g)&(perm_g==1))
        correct_avg.append(correct/num_nodes)

    return np.mean(correct_avg)

