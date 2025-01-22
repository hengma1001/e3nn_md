"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of february 2021
"""

import math
from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import ExtractIr, FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from torch import nn

from .diffusion import diffusion_sampler


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        self.embeddings = torch.exp(torch.arange(half_dim) * -embeddings)

    def forward(self, time):
        device = time.device
        embeddings = time[:, None] * self.embeddings[None, :].to(device)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


def radius_graph(pos, r_max, batch) -> torch.Tensor:
    # naive and inefficient version of torch_cluster.radius_graph
    r = torch.cdist(pos, pos)
    index = ((r < r_max) & (r > 0)).nonzero().T
    index = index[:, batch[index[0]] == batch[index[1]]]
    return index


@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_out : `e3nn.o3.Irreps` or None
        representation of the output node features

    number_of_edge_features : int
        number of scalar (0e) features of the edge used to feed the FC network

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_out: Optional[o3.Irreps],
        number_of_edge_features: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_out
        )

        self.lin1 = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_node_attr, self.irreps_in
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [number_of_edge_features]
            + radial_layers * [radial_neurons]
            + [tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(
            irreps_mid, self.irreps_node_attr, self.irreps_out
        )

    def forward(
        self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_features
    ) -> torch.Tensor:
        weight = self.fc(edge_features)

        x = node_input

        s = self.sc(x, node_attr)
        x = self.lin1(x, node_attr)

        edge_features = self.tp(x[edge_src], edge_attr, weight)
        x = scatter(edge_features, edge_dst, dim_size=x.shape[0]).div(
            self.num_neighbors**0.5
        )

        x = self.lin2(x, node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x


def smooth_cutoff(x):
    u = 2 * (x - 1)
    y = (math.pi * u).cos().neg().add(1).div(2)
    y[u > 0] = 0
    y[u < -1] = 1
    return y


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class Network(torch.nn.Module):
    r"""equivariant neural network

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features

    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features

    irreps_out : `e3nn.o3.Irreps`
        representation of the output features

    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials

    layers : int
        number of gates (non linearities)

    max_radius : float
        maximum radius for the convolution

    number_of_basis : int
        number of basis on which the edge length are projected

    radial_layers : int
        number of hidden layers in the radial fully connected network

    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network

    num_neighbors : float
        typical number of nodes at a distance ``max_radius``

    num_nodes : float
        typical number of nodes in a graph
    """

    def __init__(
        self,
        irreps_in: o3.Irreps | str,
        irreps_hidden: o3.Irreps | str,
        irreps_out: o3.Irreps | str,
        irreps_node_attr: o3.Irreps | str,
        irreps_edge_attr: o3.Irreps | int,
        layers: int,
        max_radius: float,
        number_of_basis: int,
        radial_layers: int,
        radial_neurons: int,
        num_neighbors: float,
        num_nodes: float,
        node_attr_n_kind: Optional[int] = None,
        node_attr_emb_dim: Optional[int] = None,
        time_emb_dim: Optional[int] = None,
        radius_decay: Optional[float] = None,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.node_attr_n_kind = node_attr_n_kind
        self.node_attr_emb_dim = node_attr_emb_dim
        self.time_emb_dim = time_emb_dim
        self.radius_decay = radius_decay
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)

        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr)
            if irreps_node_attr is not None
            else o3.Irreps("0e")
        )

        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(irreps_edge_attr)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        self.ext_z = ExtractIr(self.irreps_node_attr, "0e")
        number_of_edge_features = number_of_basis + 2 * self.irreps_node_attr.count(
            "0e"
        )

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        if self.time_emb_dim:
            dim = self.time_emb_dim
            time_dim = dim * 2
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, 3 * 2),  # compress time emb to 3D
            )
            # self.pos_mlp = nn.Sequential(
            #     nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim)
            # )

        if self.node_attr_emb_dim:
            self.node_attr_emb = nn.Embedding(
                self.node_attr_n_kind, self.node_attr_emb_dim
            )

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            irreps = gate.irreps_out
            self.layers.append(Compose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        time: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        radius_decay_ratio: Optional[float] = None,
        use_original_geometry: bool = False,
    ) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        pos : `torch.Tensor` (optional)
            prio positions
        time : `torch.Tensor` (optional)
        """
        if pos is None:
            pos = data["pos"]

        if "batch" in data:
            batch = data["batch"]
        else:
            batch = pos.new_zeros(pos.shape[0], dtype=torch.long)

        if time is not None and radius_decay_ratio is not None:
            max_radius = float(self.max_radius * radius_decay_ratio)
        else:
            max_radius = self.max_radius

        if use_original_geometry:
            edge_index = radius_graph(data["pos"], max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]
        else:
            edge_index = radius_graph(pos, max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = pos[edge_src] - pos[edge_dst]
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        edge_attr = smooth_cutoff(edge_length / max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        elif time is not None:
            t = self.time_mlp(time)
            scale, shift = t.chunk(2, dim=1)
            # x = self.pos_mlp(pos)
            x = pos * (1 + scale) + shift  # use position as attr
            # x = x.reshape((x.shape.numel(), 1))
        else:
            assert self.irreps_in is None
            x = pos.new_ones((pos.shape[0], 1))

        if self.input_has_node_attr and "z" in data and self.node_attr_emb_dim:
            z = self.node_attr_emb(data["z"])
            z = torch.squeeze(z)
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = pos.new_ones((pos.shape[0], 1))

        scalar_z = self.ext_z(z)
        edge_features = torch.cat(
            [edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1
        )

        for lay in self.layers:
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_features)
            # if t is not None:
            #     x = x * (1 + scale) + shift

        if self.reduce_output:
            return scatter(x, batch, dim_size=int(batch.max()) + 1).div(
                self.num_nodes**0.5
            )
        else:
            return x


class e3_diffusion(L.LightningModule):
    def __init__(
        self,
        time_step: int,
        scheduler: "str",
        weight_fill: float = 0.005,
        **model_kwargs: Dict,
    ) -> None:
        """_summary_

        Parameters
        ----------
        time_step : int
            _description_
        scheduler : str
            _description_
        weight_fill : float, optional
            _description_, by default 0.005
        """
        super().__init__()
        self.save_hyperparameters()

        self.time_step = time_step
        self.diffu_sampler = diffusion_sampler(time_step, scheduler)
        self.weight_fill = weight_fill
        self.model = Network(**model_kwargs)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        time: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        use_original_geometry: bool = False,
    ) -> torch.Tensor:
        """evaluate the network

        Parameters:
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        time : `torch.Tensor` (optional)
        """
        # if time is not None:
        #     rdr = 1 - self.diffu_sampler.betas[int(time[0].to("cpu"))] * 4
        # else:
        #     rdr = None
        results = self.model(data, time, pos, use_original_geometry)  # , rdr)
        return results

    def _get_loss(self, data, time=None, noise=None):
        device = self.device
        pos = data["pos"]
        if noise is None:
            noise = torch.randn_like(pos)

        if time is None:
            time = torch.randint(self.time_step, (1,))
        else:
            time = torch.Tensor(time).to(device)

        # edge_index = radius_graph(
        #     pos,
        #     self.model.max_radius,
        #     batch=data["batch"],
        # )
        x_noisy = self.diffu_sampler.q_sample(pos, int(time[0]), noise=noise)
        predicted_noise = self(
            data.to(device),
            time.to(device),
            x_noisy.to(device),
            use_original_geometry=True,
        )
        self.log("sample time", int(time[0]))
        return F.smooth_l1_loss(noise, predicted_noise)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "validation_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("validation_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss)
        return loss
