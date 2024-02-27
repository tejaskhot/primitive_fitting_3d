import os
import sys
import numpy as np
import torch
from torch.autograd import Variable

NUM_CATEGORY = 13
NUM_GROUPS = 3


def var(x, cuda=True):
    if cuda:
        return Variable(torch.Tensor(x).cuda())
    else:
        return Variable(torch.Tensor(x))


def countunique(A, Amax):
    res = np.empty(A.shape[1:], A.dtype)
    c = np.empty(Amax + 1, A.dtype)
    for i in range(A.shape[1]):
        for j in range(A.shape[2]):
            T = A[:, i, j]
            for k in range(c.size):
                c[k] = 0
            for x in T:
                c[x] = 1
            res[i, j] = c.sum()
    return res


def exp_lr_scheduler(
    optimizer, global_step, init_lr, decay_steps, decay_rate, lr_clip, staircase=True
):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if staircase:
        lr = init_lr * decay_rate ** (global_step // decay_steps)
    else:
        lr = init_lr * decay_rate ** (global_step / decay_steps)
    lr = max(lr, lr_clip)

    if global_step % decay_steps == 0:
        print("LR is set to {}".format(lr))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def convert_seg_to_one_hot(labels, no_batch=False):
    # labels:BxN
    if no_batch:
        labels = np.expand_dims(labels, axis=0)
    label_one_hot = np.zeros((labels.shape[0], labels.shape[1], NUM_CATEGORY))
    pts_label_mask = np.zeros((labels.shape[0], labels.shape[1]))

    un, cnt = np.unique(labels, return_counts=True)
    label_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in label_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(labels.shape[0]):
        for jdx in range(labels.shape[1]):
            if labels[idx, jdx] != -1:
                label_one_hot[idx, jdx, labels[idx, jdx]] = 1
                pts_label_mask[idx, jdx] = float(totalnum) / float(
                    label_count_dictionary[labels[idx, jdx]]
                )  # 1. - float(label_count_dictionary[labels[idx, jdx]]) / totalnum

    return label_one_hot.astype(np.float32), pts_label_mask.astype(np.float32)


def convert_groupandcate_to_one_hot(grouplabels, no_batch=False, num_groups=NUM_GROUPS):
    # grouplabels: BxN
    if no_batch:
        grouplabels = np.expand_dims(grouplabels, axis=0)
    group_one_hot = np.zeros((grouplabels.shape[0], grouplabels.shape[1], num_groups))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))
    # group_counts = np.zeros((grouplabels.shape[0]))
    group_counts = []

    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in group_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        # group_counts[idx] = len(un)
        group_counts.append(len(un))
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                group_one_hot[
                    idx, jdx, grouplabel_dictionary[grouplabels[idx, jdx]]
                ] = 1
                pts_group_mask[idx, jdx] = float(totalnum) / float(
                    group_count_dictionary[grouplabels[idx, jdx]]
                )  # 1. - float(group_count_dictionary[grouplabels[idx, jdx]]) / totalnum

    return (
        group_one_hot.astype(np.float32),
        pts_group_mask.astype(np.float32),
        np.array(group_counts),
    )


def get_mask(labels, grouplabels):
    # labels:BxN
    # grouplabels: BxN
    pts_label_mask = np.zeros((labels.shape[0], labels.shape[1]))
    pts_group_mask = np.zeros((grouplabels.shape[0], grouplabels.shape[1]))
    ####### labels #######
    un, cnt = np.unique(labels, return_counts=True)
    label_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in label_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(labels.shape[0]):
        for jdx in range(labels.shape[1]):
            if labels[idx, jdx] != -1:
                pts_label_mask[idx, jdx] = float(totalnum) / float(
                    label_count_dictionary[labels[idx, jdx]]
                )
    ####### group labels #######
    un, cnt = np.unique(grouplabels, return_counts=True)
    group_count_dictionary = dict(zip(un, cnt))
    totalnum = 0
    for k_un, v_cnt in group_count_dictionary.items():
        if k_un != -1:
            totalnum += v_cnt

    for idx in range(grouplabels.shape[0]):
        un = np.unique(grouplabels[idx])
        grouplabel_dictionary = dict(zip(un, range(len(un))))
        for jdx in range(grouplabels.shape[1]):
            if grouplabels[idx, jdx] != -1:
                pts_group_mask[idx, jdx] = float(totalnum) / float(
                    group_count_dictionary[grouplabels[idx, jdx]]
                )

    return pts_label_mask, pts_group_mask


def generate_group_mask(pts, grouplabels, labels):
    # grouplabels: BxN
    # pts: BxNx6
    # labels: BxN

    group_mask = np.zeros(
        (grouplabels.shape[0], grouplabels.shape[1], grouplabels.shape[1])
    )

    for idx in range(grouplabels.shape[0]):
        for jdx in range(grouplabels.shape[1]):
            for kdx in range(grouplabels.shape[1]):
                if labels[idx, jdx] == labels[idx, kdx]:
                    group_mask[idx, jdx, kdx] = 2.0

                if (
                    np.linalg.norm(
                        (pts[idx, jdx, :3] - pts[idx, kdx, :3])
                        * (pts[idx, jdx, :3] - pts[idx, kdx, :3])
                    )
                    < 0.04
                ):
                    if labels[idx, jdx] == labels[idx, kdx]:
                        group_mask[idx, jdx, kdx] = 5.0
                    else:
                        group_mask[idx, jdx, kdx] = 2.0

    return group_mask


if __name__ == "__main__":
    input_list = "data/train_hdf5_file_list.txt"
    train_file_list = provider.getDataFiles(input_list)
    num_train_file = len(train_file_list)
    train_file_idx = np.arange(0, len(train_file_list))
    np.random.shuffle(train_file_idx)
    ## load all data into memory
    all_data = []
    all_group = []
    all_seg = []
    for i in range(num_train_file):
        cur_train_filename = train_file_list[train_file_idx[i]]
        cur_data, cur_group, _, cur_seg = (
            provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)
        )
        all_data += [cur_data]
        all_group += [cur_group]
        all_seg += [cur_seg]

    all_data = np.concatenate(all_data, axis=0)
    all_group = np.concatenate(all_group, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)

    num_data = all_data.shape[0]
