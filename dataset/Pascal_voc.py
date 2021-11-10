import os
import sys
import random
import pickle
import torch
import scipy.io
import xml.dom.minidom
import numpy as np

from scipy import spatial
from pathlib import Path
from torch.utils.data import Dataset,DataLoader

from utils.config import cfg
import xml.etree.ElementTree as ET

anno_path = cfg.VOC2011.KPT_ANNO_DIR
img_path = cfg.VOC2011.ROOT_DIR + 'JPEGImages'
ori_anno_path = cfg.VOC2011.ROOT_DIR + 'Annotations'
set_path = cfg.VOC2011.SET_SPLIT
cache_path = cfg.CACHE_PATH

classes= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
          'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

VOC_ANNOTATION_ROOT = "data/PascalVoc/annotations"
VOC_FEATURE_ROOT = "data/KPT_features/PascalVoc"
VOC_IMAGE_ROOT = "data/PascalVoc/JPEGImages"

class PascalVOC(Dataset):
    def __init__(self, sets):
        """
        :param sets: 'train' or 'test'
        """
        self.sets = sets
        self.classes=classes
        self.ori_anno_path=Path(ori_anno_path)
        self.rand=np.random.RandomState(seed=cfg.RANDOM_SEED)
        assert sets == 'train' or 'test', 'No match found for dataset {}'.format(sets)
        cache_name = 'voc_db_' + sets + '.pkl'
        self.cache_path = Path(cache_path)
        self.cache_file = self.cache_path / cache_name
        if self.cache_file.exists():
            with self.cache_file.open(mode='rb') as f:
                self.xml_list = pickle.load(f)
            print('xml list loaded from {}'.format(self.cache_file))
        else:
            print('Caching xml list to {}...'.format(self.cache_file))
            self.cache_path.mkdir(exist_ok=True, parents=True)
            with np.load(set_path, allow_pickle=True) as f:
                self.xml_list = f[sets]
            before_filter = sum([len(k) for k in self.xml_list])
            self.filter_list()
            after_filter = sum([len(k) for k in self.xml_list])
            with self.cache_file.open(mode='wb') as f:
                pickle.dump(self.xml_list, f)
            print('Filtered {} images to {}. Annotation saved.'.format(before_filter, after_filter))

    def filter_list(self):
        """
        Filter out 'truncated', 'occluded' and 'difficult' images following the practice of previous works.
        In addition, this dataset has uncleaned label (in person category). They are omitted as suggested by README.
        """
        for cls_id in range(len(self.classes)):
            to_del = []
            for xml_name in self.xml_list[cls_id]:
                xml_comps = xml_name.split('/')[-1].strip('.xml').split('_')
                ori_xml_name = '_'.join(xml_comps[:-1]) + '.xml'
                voc_idx = int(xml_comps[-1])
                xml_file = self.ori_anno_path / ori_xml_name
                assert xml_file.exists(), '{} does not exist.'.format(xml_file)
                tree = ET.parse(xml_file.open())
                root = tree.getroot()
                obj = root.findall('object')[voc_idx - 1]

                difficult = obj.find('difficult')
                if difficult is not None: difficult = int(difficult.text)
                occluded = obj.find('occluded')
                if occluded is not None: occluded = int(occluded.text)
                truncated = obj.find('truncated')
                if truncated is not None: truncated = int(truncated.text)
                if difficult or occluded or truncated:
                    to_del.append(xml_name)
                    continue

                # Exclude uncleaned images
                if self.classes[cls_id] == 'person' and int(xml_comps[0]) > 2008:
                    to_del.append(xml_name)
                    continue

            for x in to_del:
                self.xml_list[cls_id].remove(x)

    def get_xml_pair(self, cls=None):
        if cls is None:
            cls = random.randrange(0, len(self.classes))
        elif type(cls) == str:
            cls = self.classes.index(cls)
        assert type(cls) == int and 0 <= cls < len(self.classes)

        return random.sample(self.xml_list[cls], 2)

    def __len__(self):
        if self.sets=="train":
            len_total=0
            for xmls in self.xml_list:
                len_total+=(len(xmls)**2)
        else:
            len_total=20000

        return len_total

    def _load_feature(self,file):
        mat_file = VOC_FEATURE_ROOT + file.split("annotations")[-1][:-4] + '.mat'
        infos = scipy.io.loadmat(mat_file)

        pts_fea = infos['pts_features']
        n_pts, dim_feat = pts_fea.shape
        if n_pts == 0:
            return None
        pts_fea = pts_fea.reshape(-1, int(cfg.MEAN_POOLING_INTERVAL_PASCAL))
        pts_fea = np.mean(pts_fea, axis=1)
        pts_fea = pts_fea.reshape((n_pts, -1))
        return pts_fea

    def _load_annotation(self,file):
        dom = xml.dom.minidom.parse(file)
        root = dom.documentElement

        image = root.getElementsByTagName('image')[0].firstChild.data

        keypoints = root.getElementsByTagName('keypoints')[0]
        kps = keypoints.getElementsByTagName('keypoint')

        annoName = []
        annoPts = []
        for kp in kps:
            x = float(kp.getAttribute('x'))
            y = float(kp.getAttribute('y'))
            name = kp.getAttribute('name')
            annoName.append(name)
            annoPts.append([x, y])

        annoName = np.array(annoName)
        annoPts = np.array(annoPts)

        return image, annoName, annoPts

    def _load_keypoint_features(self, annoFile):

        # gen image patches for nodes
        imgfile, anno_names, anno_pts = self._load_annotation(annoFile)
        imgPath = "{}/{}.jpg".format(VOC_IMAGE_ROOT, imgfile)

        # strip space in names
        for i in range(anno_names.shape[0]):
            anno_names[i] = anno_names[i].strip()
        descs = self._load_feature(annoFile)

        return anno_names, anno_pts, descs, imgPath

    def _remove_unmatched_points(self,anno_names0, anno_pts0, anno_descs0,
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

        while True:
            if self.sets=="train":
                xml_files = self.get_xml_pair(self.rand.choice(len(self.classes)))
            else:
                xml_files = self.get_xml_pair(int(index/1000))

            xml_file0 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[0])
            xml_file1 = "{}/{}".format(VOC_ANNOTATION_ROOT, xml_files[1])

            anno_names0, anno_pts0, anno_descs0, imgPath0 = self._load_keypoint_features(xml_file0)
            anno_names1, anno_pts1, anno_descs1, imgPath1 = self._load_keypoint_features(xml_file1)

            if anno_pts0.shape[0] < 3 or anno_pts1.shape[0] < 3:
                continue

            # remove unmatched points
            anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1 = self._remove_unmatched_points(
                anno_names0, anno_pts0, anno_descs0, anno_names1, anno_pts1, anno_descs1)

            if anno_pts0.shape[0] >= 3 and anno_pts1.shape[0] >= 3:
                break

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