import os
import scipy.io
import numpy as np

from scipy import spatial
from torch.utils.data import Dataset

from utils.config import cfg

WILLOW_FEATURE_ROOT = "data/KPT_features/Willow"
WILLOW_FILE_ROOT = "data/WILLOW"
WILLOW_TRAIN_NUM = 20
WILLOW_TRAIN_OFFSET = 0

categories = ["Car", "Duck", "Face", "Motorbike", "Winebottle"]

class Willow(Dataset):
    def __init__(self, sets):
        """
        :param sets: 'train' or 'test'
        """
        self.sets = sets
        self.rand = np.random.RandomState(seed=cfg.RANDOM_SEED)
        self.imgFiles = dict()
        for category in categories:
            imgs = []
            path = os.path.join(WILLOW_FILE_ROOT, category)
            for file in os.listdir(path):
                filepath = os.path.join(path, file)
                if os.path.isfile(filepath):
                    if os.path.basename(filepath).endswith('.png'):
                        imgs.append(filepath)

            if self.sets == "train":
                self.imgFiles[category] = imgs[:WILLOW_TRAIN_NUM]
            else:
                self.imgFiles[category] = imgs[WILLOW_TRAIN_NUM:]

    def __len__(self):
        if self.sets == "train":
            len_total = 0
            for category, imgs in self.imgFiles.items():
                len_total += (len(imgs) ** 2)
        else:
            len_total = 5000
        return len_total

    def _load_feature(self, file):
        mat_file = WILLOW_FEATURE_ROOT + file.split("WILLOW")[-1][:-4] + '.mat'
        infos = scipy.io.loadmat(mat_file)

        pts_fea = infos['pts_features']
        n_pts, dim_feat = pts_fea.shape
        if n_pts == 0:
            return None
        pts_fea = pts_fea.reshape(-1, int(cfg.MEAN_POOLING_INTERVAL_WILLOW))
        pts_fea = np.mean(pts_fea, axis=1)
        pts_fea = pts_fea.reshape((n_pts, -1))
        return pts_fea

    def _load_annotation(self, file):
        iLen = len(file)
        raw_file = file[:iLen - 4]
        anno_file = raw_file + ".mat"
        anno = scipy.io.loadmat(anno_file)
        pts = np.transpose(anno["pts_coord"])
        return pts

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

    def _read_image_features(self, file):
        anno_pts = self._load_annotation(file)
        descs = self._load_feature(file)

        return anno_pts, descs

    def __getitem__(self, index):

        if self.sets == "train":
            category_id = self.rand.choice(len(categories))
            category = categories[category_id]
        else:
            category = categories[int(index // 1000)]

        imgFiles = self.rand.choice(self.imgFiles[category], size=2)

        anno_pts0, anno_descs0 = self._read_image_features(imgFiles[0])
        anno_pts1, anno_descs1 = self._read_image_features(imgFiles[1])

        pts0 = anno_pts0.copy()
        pts1 = anno_pts1.copy()
        descs0 = anno_descs0.copy()
        descs1 = anno_descs1.copy()

        index0 = np.arange(0, pts0.shape[0])
        self.rand.shuffle(index0)
        pts0 = pts0[index0]
        descs0 = descs0[index0]

        index1 = np.arange(0, pts1.shape[0])
        self.rand.shuffle(index1)
        pts1 = pts1[index1]
        descs1 = descs1[index1]

        # normalize point coordinates
        pts0 = self._normalize_coordinates(pts0)
        pts1 = self._normalize_coordinates(pts1)

        # record ground-truth matches
        gX = np.zeros((pts0.shape[0], pts1.shape[0]))
        for i in range(anno_pts0.shape[0]):
            gX[i][i] = 1.0
        gX = np.transpose(np.transpose(gX[index0])[index1])

        return descs0, descs1, pts0, pts1, gX,
