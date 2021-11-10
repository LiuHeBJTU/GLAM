from utils.compute_accuracy import accuracy

categories=["Car","Duck","Face","Motorbike","Winebottle"]

def willow_test(model,dataset_test):
    acc_dict=dict()
    for category in categories:
        acc_dict[category]=[]

    model.eval()
    for idx , data in enumerate(dataset_test):
        descs0, descs1, pts0, pts1, X, atten_mask, key_mask, num_node_list = data
        match_matrix = model(descs0, descs1, pts0, pts1, atten_mask, key_mask)
        acc=accuracy(match_matrix.data.cpu().numpy(), X.cpu().numpy(), num_node_list)
        acc_dict[categories[int(idx/1000)]].append(acc)

    return acc_dict