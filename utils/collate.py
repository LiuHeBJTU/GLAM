import torch
import numpy as np

def collate_fn(list_data):
    batchsize=len(list_data)
    descs_batch_0, descs_batch_1, pts_batch_0, pts_batch_1 ,X_batch, num_node_list=[], [], [], [], [],[]
    max_size=0
    for data in list_data:
        descs0, descs1, pts0, pts1,X=data
        if X.shape[0]>max_size:
            max_size=X.shape[0]

    atten_mask=torch.zeros((batchsize,max_size,max_size))
    key_mask=torch.zeros((batchsize,max_size))
    for idx in range(batchsize):
        data= list_data[idx]
        descs0, descs1, pts0, pts1, X = data
        num_nodes=descs0.shape[0]
        descs0_padded=np.pad(descs0,((0,max_size-num_nodes),(0,0)),mode="constant",constant_values = (0,0))
        descs1_padded = np.pad(descs1, ((0, max_size - num_nodes), (0, 0)), mode="constant",constant_values=(0, 0))
        pts0_padded = np.pad(pts0, ((0, max_size - num_nodes), (0, 0)), mode="constant",constant_values=(0, 0))
        pts1_padded = np.pad(pts1, ((0, max_size - num_nodes), (0, 0)), mode="constant",constant_values=(0, 0))
        X_padded = np.pad(X, ((0, max_size - num_nodes), (0, max_size - num_nodes)), mode="constant",constant_values=(0, 0))

        descs_batch_0.append(np.expand_dims(descs0_padded,axis=1))
        descs_batch_1.append(np.expand_dims(descs1_padded,axis=1))
        pts_batch_0.append(np.expand_dims(pts0_padded,axis=1))
        pts_batch_1.append(np.expand_dims(pts1_padded,axis=1))
        X_batch.append(np.expand_dims(X_padded,axis=0))
        num_node_list.append(num_nodes)

        atten_mask[idx,:num_nodes,:num_nodes]= 1
        key_mask[idx,:num_nodes]=1

        # atten_mask[idx, (num_nodes-1):, :] = 0
        # atten_mask[idx, :, (num_nodes - 1):] = 0
        # key_mask[idx, (num_nodes-1):] = 0

    return torch.from_numpy(np.concatenate(descs_batch_0,axis=1)).type(torch.float),torch.from_numpy(np.concatenate(descs_batch_1,axis=1)).type(torch.float),\
           torch.from_numpy(np.concatenate(pts_batch_0,axis=1)).type(torch.float),torch.from_numpy(np.concatenate(pts_batch_1,axis=1)).type(torch.float),\
           torch.from_numpy(np.concatenate(X_batch,axis=0)).type(torch.float), atten_mask.type(torch.float), key_mask.type(torch.float),num_node_list