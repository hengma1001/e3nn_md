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

from e3nn_md.esm_embed.embedding import ESM_embedder


def parse_traj(traj_path, top_path, embedder=True, sel="protein and not name H*"):
    data = {}
    data["sys_name"] = os.path.basename(top_path)

    mda_u = mda.Universe(top_path, traj_path)
    atmgrp_selected = mda_u.select_atoms(sel)
    data["res_atom_name"] = [atom.resname + atom.name for atom in atmgrp_selected.atoms]
    if embedder:
        embedder = ESM_embedder()
        esm_embeddings = embedder.cal_embedding(top_path)
        data["esm_embedding"] = [esm_embeddings[atom.resindex] for atom in atmgrp_selected.atoms]

    positions = np.zeros((mda_u.trajectory.n_frames, atmgrp_selected.n_atoms, 3))
    for ts in mda_u.trajectory:
        positions[ts.frame] = (atmgrp_selected.positions - atmgrp_selected.center_of_mass()) / 10  # convert to nm
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

    data = dbs[0]
    topology = torch_geometric.data.Data(
        z=torch.from_numpy(label_encoder.transform(data["res_atom_name"])).reshape(-1, 1),
        esm=torch.stack(data["esm_embedding"]),
    )

    dbs_refined = []
    for data in tqdm(dbs):
        for i in tqdm(range(len(data["pos"]) - 1)):
            frame_data = torch_geometric.data.Data(
                pos=data["pos"][i],
                y=kabsch_torch(data["pos"][i + 1], data["pos"][i]) - data["pos"][i],
                sys_name=data["sys_name"],
            )
            dbs_refined.append(frame_data)
    return topology, dbs_refined, full_voca_size, label_encoder


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


def write_pdb(pdb_file, positions, output_pdb, sel_str="protein and name CA"):
    mda_u = mda.Universe(pdb_file)
    atmgrp_sel = mda_u.select_atoms(sel_str)
    atmgrp_sel.positions = positions
    atmgrp_sel.write(output_pdb)


def kabsch_torch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    return torch.matmul(p, R.transpose(0, 1))
    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return R, t, rmsd


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
