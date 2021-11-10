import torch
import numpy as np

def entropy_loss(perm_pred,perm_gt):
    if torch.cuda.is_available():
        perm_gt=perm_gt.cuda()
    loss_all = 5.0*perm_gt*torch.log(perm_pred+(1e-3))+(1-perm_gt)*torch.log(1-perm_pred+(1e-3))

    loss= torch.mean(-loss_all)

    return loss

def loss_(outputs,targets,num_nodes_list):

    loss,num_nodes_all=0,0
    for idx in range(len(num_nodes_list)):
        num_nodes=num_nodes_list[idx]
        output= outputs[idx][:num_nodes,:num_nodes]
        target= targets[idx][:num_nodes,:num_nodes]
        num_nodes_all=num_nodes_all+num_nodes

        loss= loss+entropy_loss(output,target)

    loss = loss/len(num_nodes_list)

    return loss