import json
import random
import scipy.io
import numpy as np

from scipy import spatial
from utils.config import cfg
from torch.utils.data import Dataset

Spair_ANNOTATION_ROOT = "data/SPair71k/ImageAnnotation"
Spair_FEATURE_ROOT = "data/KPT_features/SPair71k"
Spair_IMAGE_ROOT = "data/SPair71k/JPEGImages"
Spair_PAIRANNOTATION_ROOT = "data/SPair71k/PairAnnotation"

CATEGORIES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']

train_anno_txt = "data/SPair71k/Layout/large/trn.txt"
test_anno_txt = "data/SPair71k/Layout/large/test.txt"


class Spair(Dataset):
    def __init__(self, sets):
        """
        :param sets: 'train' or 'test'
        """
        self.sets = sets
        self.classes = CATEGORIES
        self.rand = np.random.RandomState(seed=cfg.RANDOM_SEED)

        self.dataset_dict = dict()
        if self.sets == "train":
            anno_list = open(train_anno_txt, "r").read().split("\n")[:-1]
        else:
            anno_list = open(test_anno_txt, "r").read().split("\n")[:-1]

        for anno_name in anno_list:
            cls_name = anno_name.split(":")[-1]
            srcImg_name = anno_name.split("-")[1]
            dstImg_name = anno_name.split("-")[-1].split(":")[0]
            if cls_name not in self.dataset_dict.keys():
                self.dataset_dict[cls_name] = [(srcImg_name, dstImg_name)]
            else:
                self.dataset_dict[cls_name].append((srcImg_name, dstImg_name))

        self.num_pairs_total = 0
        self.num_pairs_list = []
        for key, value in self.dataset_dict.items():
            self.num_pairs_list.append(self.num_pairs_total + len(value))
            self.num_pairs_total = self.num_pairs_total + len(value)

    def __len__(self):

        return self.num_pairs_total

    def _load_feature(self, file):
        mat_file = Spair_FEATURE_ROOT + file.split("ImageAnnotation")[-1][:-5] + '.mat'
        infos = scipy.io.loadmat(mat_file)

        pts_fea = infos['pts_features']
        n_pts, dim_feat = pts_fea.shape
        if n_pts == 0:
            return None
        pts_fea = pts_fea.reshape(-1, int(cfg.MEAN_POOLING_INTERVAL_PASCAL))
        pts_fea = np.mean(pts_fea, axis=1)
        pts_fea = pts_fea.reshape((n_pts, -1))
        return pts_fea

    def _load_annotation(self, file):
        with open(file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)

        image = json_data["filename"][:-4]
        annoName = []
        annoPts = []
        for key, value in json_data["kps"].items():
            if value is not None:
                annoName.append(key)
                annoPts.append(value)
        annoName = np.array(annoName)
        annoPts = np.array(annoPts)

        return image, annoName, annoPts

    def _load_keypoint_features(self, annoFile):

        # gen image patches for nodes
        imgfile, anno_names, anno_pts = self._load_annotation(annoFile)
        imgPath = "{}/{}.jpg".format(Spair_IMAGE_ROOT, imgfile)

        # strip space in names
        for i in range(anno_names.shape[0]):
            anno_names[i] = anno_names[i].strip()

        descs = self._load_feature(annoFile)

        return anno_names, anno_pts, descs, imgPath

    def _remove_unmatched_points(self, anno_names0, anno_pts0, anno_descs0,
                                 anno_names1, anno_pts1, anno_descs1):
        valid0 = np.zeros(anno_names0.shape[0], dtype=np.bool)
        valid1 = np.zeros(anno_names1.shape[0], dtype=np.bool)

        for i in range(anno_names0.shape[0]):
            for k in range(anno_names1.shape[0]):
                if anno_names0[i] == anno_names1[k]:
                    valid0[i] = True
                    valid1[k] = True
                    break

        anno_names0 = anno_names0[valid0]
        anno_pts0 = anno_pts0[valid0]
        anno_descs0 = anno_descs0[valid0]

        anno_names1 = anno_names1[valid1]
        anno_pts1 = anno_pts1[valid1]
        anno_descs1 = anno_descs1[valid1]

        return anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1

    def _normalize_coordinates(self, points):
        # normalize by center
        center = np.sum(points, axis=0) / points.shape[0]
        norm_points = np.transpose(points)
        norm_points[0] = norm_points[0] - center[0]
        norm_points[1] = norm_points[1] - center[1]

        # normalized by max_distance
        distance = spatial.distance.cdist(points, points)
        maxDst = np.max(distance)
        norm_points = norm_points / maxDst

        points = np.transpose(norm_points)

        return points

    def __getitem__(self, index):

        unavailable = False
        while True:
            if self.sets == "train":
                category_id = self.rand.choice(len(self.classes))
                category = self.classes[category_id]
                pair_names = random.choice(self.dataset_dict[category])
            else:
                if unavailable:
                    index += 1
                num_pairs_temp = self.num_pairs_list + [index]
                sorted_id = sorted(range(len(num_pairs_temp)), key=lambda k: num_pairs_temp[k], reverse=False)
                location_id = sorted_id.index(18)
                category = self.classes[location_id]
                pair_names = self.dataset_dict[category][self.num_pairs_list[location_id] - (index + 1)]

                # print(index, location_id,category, self.num_pairs_list[location_id]-(index+1))
            xml_name0, xml_name1 = pair_names

            xml_file0 = Spair_ANNOTATION_ROOT + "/" + category + "/" + xml_name0 + ".json"
            xml_file1 = Spair_ANNOTATION_ROOT + "/" + category + "/" + xml_name1 + ".json"

            anno_names0, anno_pts0, anno_descs0, imgPath0 = self._load_keypoint_features(xml_file0)
            anno_names1, anno_pts1, anno_descs1, imgPath1 = self._load_keypoint_features(xml_file1)

            # if anno_pts0.shape[0] < 3 or anno_pts1.shape[0] < 3:
            #     continue
            # remove unmatched points
            anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1 = self._remove_unmatched_points(
                anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1)

            if anno_pts0.shape[0] >= 2 and anno_pts1.shape[0] >= 2:
                break
            else:
                unavailable = True

        pts0 = anno_pts0.copy()
        pts1 = anno_pts1.copy()
        descs0 = anno_descs0.copy()
        descs1 = anno_descs1.copy()
        names0 = anno_names0.copy()
        names1 = anno_names1.copy()

        index1 = np.arange(0, pts1.shape[0])
        self.rand.shuffle(index1)
        pts1 = pts1[index1]
        descs1 = descs1[index1]
        names1 = names1[index1]

        # normalize point coordinates
        pts0 = self._normalize_coordinates(pts0)
        pts1 = self._normalize_coordinates(pts1)

        # record ground-truth matches
        gX = np.zeros((pts0.shape[0], pts1.shape[0]))
        for i in range(anno_pts0.shape[0]):
            for k in range(anno_pts1.shape[0]):
                if names0[i] == names1[k]:
                    gX[i][k] = 1.0
                    break

        return descs0, descs1, pts0, pts1, gX
