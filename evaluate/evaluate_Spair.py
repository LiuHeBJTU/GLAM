from dataset.Spair import Spair
from utils.compute_accuracy import accuracy

categories= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
dataset_spair= Spair(sets="test")
num_pairs_list= dataset_spair.num_pairs_list

def spair_test(model,dataset_test):
    acc_dict=dict()
    for category in categories:
        acc_dict[category]=[]

    model.eval()
    for idx , data in enumerate(dataset_test):
        descs0, descs1, pts0, pts1, X, atten_mask, key_mask, num_node_list = data
        match_matrix = model(descs0, descs1, pts0, pts1, atten_mask, key_mask)
        acc=accuracy(match_matrix.data.cpu().numpy(), X.cpu().numpy(), num_node_list)

        num_pairs_temp = num_pairs_list + [idx]
        sorted_id = sorted(range(len(num_pairs_temp)), key=lambda k: num_pairs_temp[k], reverse=False)
        location_id = sorted_id.index(18)
        acc_dict[categories[location_id]].append(acc)

    return acc_dict