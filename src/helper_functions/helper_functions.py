import os
import random
from copy import deepcopy

import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO

def get_only_relevant_gt(gt_labels, relevant_inds):
    if isinstance(gt_labels[0], list):
        new_samples = [s for s in
                       gt_labels if any(x in list(relevant_inds) for x in s)]
        new_indices = [i for i, s in enumerate(gt_labels) if any(x in list(relevant_inds) for x in s)]
    else:
        new_samples = [s for s in
                       gt_labels if s in relevant_inds]
        new_indices = [i for i, s in
                       enumerate(gt_labels) if s in relevant_inds]

    filtered_samples = []
    for s in new_samples:
        gt_label = [list(relevant_inds).index(x) for x in s if x in list(relevant_inds)]
        filtered_samples.append(gt_label)

    return filtered_samples, new_indices


def one_hot_to_class_labels(one_hot_array):
    samples = []
    if isinstance(one_hot_array, np.ndarray):
        for i, s in enumerate(one_hot_array):
            idx_hot = np.where(s)[0]
            samples.append(list(idx_hot))
    return samples

def get_dist(gallery, vecs, k=10, for_map=False):
    if for_map:
        mat = np.reshape(gallery, (gallery.shape[0], vecs.shape[1], -1))

        K_c = mat.shape[2]
        dot_prod = [np.expand_dims(np.matmul(vecs, mat[:, :, i].transpose(1, 0)), axis=2) for i in
                    range(K_c)]
    else:
        mat = np.reshape(vecs, (vecs.shape[0], gallery.shape[1], -1))
        K_c = mat.shape[2]
        dot_prod = [np.expand_dims(np.matmul(mat[:, :, i], gallery.transpose(1, 0)), axis=2) for i in
                    range(K_c)]
    dist = -np.max(np.concatenate(dot_prod, axis=2), axis=-1)

    if for_map:
        return dist

    ids = np.argsort(dist, axis=1)
    top_k_ids = ids[:, :k]
    top_k_dist = np.take_along_axis(dist, ids, axis=1)[:, :k]
    return top_k_ids, top_k_dist


def get_knns(gallery, vecs, k=-1, for_map=False):
    if k == -1:
        k = gallery.shape[0]  # Using all possible samples
    return get_dist(gallery, vecs, k=k, for_map=for_map)


def calc_F1(gt_labels, idxs, k, num_classes=None, relevant_inds=None):
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    class_samples = np.zeros(num_classes)
    gt_labels = one_hot_to_class_labels(gt_labels)
    if relevant_inds is not None:
        gt_labels, indices = get_only_relevant_gt(gt_labels, relevant_inds)
        idxs = idxs[indices]

    num_samples = len(gt_labels)

    for i in range(num_samples):
        gt_label = gt_labels[i]
        if isinstance(gt_label, list):
            tps = [elem in idxs[i][:k] for elem in gt_label]
            for j in range(len(gt_label)):
                TP[gt_label[j]] += tps[j]
                class_samples[gt_label[j]] += 1
            fps = [elem not in gt_label for elem in idxs[i][:k]]

            for j in range(k):
                if j < FP.shape[0]:
                    FP[idxs[i][j]] += fps[j]
        else:
            raise NotImplementedError

    TP_s = np.nansum(TP)
    FP_s = np.nansum(FP)
    precision_o = TP_s / (TP_s + FP_s)

    class_samples_s = np.nansum(class_samples)
    recall_o = TP_s / class_samples_s

    if precision_o == 0 or recall_o == 0:  # avoid nan if both zero
        F1_o = 0
    else:
        F1_o = 2 * precision_o * recall_o / (precision_o + recall_o)

    return 100*precision_o, 100*recall_o, 100*F1_o


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds, relevant_inds = None, num_classes = None):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    gt_labels = one_hot_to_class_labels(targs)
    if relevant_inds is not None:
        new_gt_labels, indices = get_only_relevant_gt(gt_labels, relevant_inds)
        preds = preds[indices]
        targs = targs[indices,:]
        targs = targs[:,relevant_inds]

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = 1-preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
