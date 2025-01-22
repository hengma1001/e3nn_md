import glob
import os
import random

import MDAnalysis as mda
import numpy as np
import torch
import torch_geometric
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_traj(traj_path, top_path, sel="protein and not name H*"):
    data = {}
    data["sys_name"] = os.path.basename(top_path)

    mda_u = mda.Universe(top_path, traj_path)
    protein_noH = mda_u.select_atoms(sel)
    data["res_atom_name"] = [atom.resname + atom.name for atom in protein_noH.atoms]

    positions = np.zeros((mda_u.trajectory.n_frames, protein_noH.n_atoms, 3))
    for ts in mda_u.trajectory:
        positions[ts.frame] = protein_noH.positions / 10  # convert to nm
    data["pos"] = torch.Tensor(positions)
    return data


def trajs_to_dbs(top_file, traj_files, **kwargs):
    dbs = [parse_traj(traj, top_file, **kwargs) for traj in tqdm(traj_files, total=len(traj_files))]
    return dbs


def dbs_to_torch(dbs):
    full_voca = np.concatenate([data["res_atom_name"] for data in dbs])
    full_voca_size = len(set(full_voca))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(full_voca)

    dbs_refined = []
    for data in tqdm(dbs):
        for j in tqdm(range(len(data["pos"]) - 1)):
            frame_data = torch_geometric.data.Data(
                pos=data["pos"][j],
                y=data["pos"][j + 1],
                z=torch.from_numpy(label_encoder.transform(data["res_atom_name"]).reshape(-1, 1)),
                sys_name=data["sys_name"],
            )
            dbs_refined.append(frame_data)
    return dbs_refined, full_voca_size, label_encoder


def dbs_split(dbs, split_ratio=[0.7, 0.2, 0.1], shuffle=True, random_seed=0, batch_size=64):
    if shuffle:
        random.seed(random_seed)
        random.shuffle(dbs)
    train, val_test = train_test_split(dbs, train_size=int(split_ratio[0] * len(dbs)))
    val, test = train_test_split(val_test, train_size=int(split_ratio[1] * len(dbs)))

    train = torch_geometric.loader.DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    val = torch_geometric.loader.DataLoader(val, batch_size=batch_size)
    test = torch_geometric.loader.DataLoader(test, batch_size=batch_size)

    return train, val, test


# def pdbs_to_datasets(comp_paths, split_ratio=[0.7, 0.2, 0.1], **kwargs):
#     dbs, full_voca_size = pdbs_to_dbs(comp_paths, **kwargs)
#     train, val_test = train_test_split(dbs, train_size=int(split_ratio[0] * len(dbs)))
#     val, test = train_test_split(val_test, train_size=int(split_ratio[1] * len(dbs)))

#     train = torch_geometric.loader.DataLoader(train, batch_size=1, shuffle=True)
#     val = torch_geometric.loader.DataLoader(val, batch_size=1, shuffle=True)
#     test = torch_geometric.loader.DataLoader(test, batch_size=1, shuffle=True)

#     return train, val, test, full_voca_size


# feat = torch.from_numpy(features)  # convert to pytorch tensors
# ys = torch.from_numpy(labels)  # convert to pytorch tensors
# traj_data = []
# distances = ys - feat  # compute distances to next frame


# # make torch_geometric dataset
# # we want this to be an iterable list
# # x = None because we have no input features
# for frame, label in zip(feat, distances):
#     traj_data += [
#         torch_geometric.data.Data(
#             x=None, pos=frame.to(torch.float32), y=label.to(torch.float32)
#         )
#     ]

# train_split = 1637
# train_loader = torch_geometric.loader.DataLoader(
#     traj_data[:train_split], batch_size=1, shuffle=False
# )

# test_loader = torch_geometric.loader.DataLoader(
#     traj_data[train_split:], batch_size=1, shuffle=False
# )
