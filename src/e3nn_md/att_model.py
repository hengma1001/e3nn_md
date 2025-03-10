"""model with self-interactions and gates

Exact equivariance to :math:`E(3)`

version of february 2021
"""

import math
from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F
from click import Option
from e3nn import o3
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from e3nn.nn import ExtractIr, FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from torch import nn
from torch_geometric.nn.pool import radius_graph
from torch_scatter import scatter


# @compile_mode("script")
class Attention(torch.nn.Module):
    """_summary_

    Parameters
    ----------
    torch : _type_
        _description_
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_query: o3.Irreps,
        irreps_key: o3.Irreps,
        irreps_value: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        # irreps_out: o3.Irreps,
        number_of_edge_features: int,
        radial_layers: int,
        radial_neurons: int,
        layer_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_value = irreps_value
        # self.irreps_out = irreps_out

        self.irreps_node_attr = irreps_node_attr
        self.irreps_edge_attr = irreps_edge_attr

        self.layer_normalize = layer_normalize

        self.sc = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_value)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)

        self.hq = o3.Linear(self.irreps_in, self.irreps_query)

        self.tp_k = FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_edge_attr, self.irreps_key, shared_weights=False
        )
        self.fc_k = FullyConnectedNet(
            [number_of_edge_features] + radial_layers * [radial_neurons] + [self.tp_k.weight_numel],
            act=torch.nn.functional.silu,
        )

        self.tp_v = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            self.irreps_value,
            shared_weights=False,
        )
        self.fc_v = FullyConnectedNet(
            [number_of_edge_features] + radial_layers * [radial_neurons] + [self.tp_v.weight_numel],
            act=torch.nn.functional.silu,
        )

        self.dot = FullyConnectedTensorProduct(self.irreps_query, self.irreps_key, "0e")

        if self.layer_normalize:
            self.norm = nn.LayerNorm(self.irreps_value.dim)

    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_sh,
        edge_features,
        edge_weight_cutoff,
    ) -> torch.Tensor:

        assert len(edge_src) == len(edge_dst)
        node_update = self.sc(node_input, node_attr)
        node_input = self.lin1(node_input, node_attr)

        q = self.hq(node_input)
        k = self.tp_k(node_input[edge_src], edge_sh, self.fc_k(edge_features))
        v = self.tp_v(node_input[edge_src], edge_sh, self.fc_v(edge_features))

        k = k.div(k.shape[-1] ** 0.5)
        # print(k.shape, q.shape, v.shape)
        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_dst], k).exp()
        z = scatter(exp, edge_dst, dim=0, dim_size=len(node_input))
        z[z == 0] = 1
        alpha = exp / z[edge_dst]
        # print(v.shape, alpha.shape)

        # node_out = node_update + scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(node_input))
        node_out = scatter(alpha.relu().sqrt() * v, edge_dst, dim=0, dim_size=len(node_input))

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        node_out = c_s * node_update + c_x * node_out

        # return node_out
        if self.layer_normalize:
            return self.norm(node_out)
        else:
            return node_out


class MultiHeadGraphAttention(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_query: o3.Irreps,
        irreps_key: o3.Irreps,
        irreps_value: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        # irreps_out: o3.Irreps,
        number_of_edge_features: int,
        radial_layers: int,
        radial_neurons: int,
        layer_normalize: bool = True,
        n_heads: int = 1,
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.head_list = torch.nn.ModuleList()

        for i in range(n_heads):
            self.head_list.append(
                Attention(
                    irreps_in,
                    irreps_query,
                    irreps_key,
                    irreps_value,  # irreps_gated,  # gate.irreps_in,  # value irreps_value,
                    irreps_node_attr,
                    irreps_edge_attr,
                    number_of_edge_features,
                    radial_layers,
                    radial_neurons,
                    layer_normalize,
                )
            )

        self.lin1 = o3.Linear(irreps_value * n_heads, irreps_value)

    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_sh,
        edge_features,
        edge_weight_cutoff,
    ) -> torch.Tensor:
        head_outputs = []
        for head in self.head_list:
            head_outputs.append(
                head(
                    node_input,
                    node_attr,
                    edge_src,
                    edge_dst,
                    edge_sh,
                    edge_features,
                    edge_weight_cutoff,
                )
            )
        x = torch.cat(head_outputs, 1)
        x = self.lin1(x)
        return x


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

    def forward(self, *input_s):
        x = self.first(*input_s)
        x = self.second(x)
        return x


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
        irreps_in: Optional[str],
        irreps_query: str,
        irreps_key: str,
        irreps_value: str,
        irreps_hidden: str,
        irreps_out: str,
        irreps_node_attr: str,
        irreps_edge_attr: int,
        n_heads: int = 1,
        layers: int = 1,
        max_radius: float = 1.0,
        number_of_basis: int = 20,
        radial_layers: int = 1,
        radial_neurons: int = 128,
        num_neighbors: float = 20,
        num_nodes: float = 20,
        node_attr_n_kind: Optional[int] = None,
        node_attr_emb_dim: Optional[int] = None,
        reduce_output: bool = True,
        use_esm_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.node_attr_n_kind = node_attr_n_kind
        self.node_attr_emb_dim = node_attr_emb_dim
        self.n_heads = n_heads
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.irreps_value = o3.Irreps(irreps_value)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr) if irreps_node_attr is not None else o3.Irreps("0e")
        if use_esm_embeddings:
            self.irreps_node_attr = self.irreps_node_attr + o3.Irreps("1280x0e")

        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(irreps_edge_attr)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        self.ext_z = ExtractIr(self.irreps_node_attr, "0e")
        number_of_edge_features = number_of_basis + 2 * self.irreps_node_attr.count("0e")

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }

        if self.node_attr_emb_dim:
            assert self.node_attr_n_kind is not None
            self.node_attr_emb = nn.Embedding(self.node_attr_n_kind, self.node_attr_emb_dim)

        self.layers = torch.nn.ModuleList()
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
            att = MultiHeadGraphAttention(
                irreps,
                self.irreps_query,
                self.irreps_key,
                gate.irreps_in,  # self.irreps_gated,  # gate.irreps_in,  # value self.irreps_value,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                number_of_edge_features,
                radial_layers,
                radial_neurons,
                n_heads,
            )
            # irreps = self.irreps_hidden
            # self.layers.append(att)
            irreps = gate.irreps_out
            self.layers.append(Compose(att, gate))

        # sc = FullyConnectedTensorProduct(
        #     irreps,
        #     irreps,
        #     self.irreps_hidden,
        # )

        self.hidden = FullyConnectedTensorProduct(
            irreps,
            self.irreps_node_attr,
            self.irreps_hidden,
        )
        self.output_layer = FullyConnectedTensorProduct(self.irreps_hidden, self.irreps_node_attr, self.irreps_out)

        # self.mlp_layers.append(nn.Linear(self.irreps_hidden.dim, self.irreps_out.dim))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        topology: Dict[str, torch.Tensor],
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """evaluate the network

        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``batch`` the graph to which the node belong, optional
        topology: `torch_geometric.data.Data` or dict
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
        """
        topology = topology.to(data["pos"].device)
        if pos is None:
            pos = data["pos"]
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = pos.new_zeros(pos.shape[0], dtype=torch.long)

        max_radius = self.max_radius

        edge_index = radius_graph(pos, max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        edge_vec = pos[edge_src] - pos[edge_dst]

        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component")
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / max_radius))

        if self.input_has_node_in:
            assert self.irreps_in is not None
            if "x" in topology:
                x = topology["x"]
            elif self.irreps_in == o3.Irreps("1o"):
                x = data["pos"]
        else:
            assert self.irreps_in is None
            x = pos.new_ones((pos.shape[0], 1))

        if self.input_has_node_attr and "z" in topology and self.node_attr_emb_dim:
            z = self.node_attr_emb(topology["z"])

            z = torch.squeeze(z)
            if "esm" in topology:
                z = torch.cat((topology["esm"], z), dim=1)
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = pos.new_ones((pos.shape[0], 1))

        z = torch.cat([z] * (len(x) // len(z)))

        scalar_z = self.ext_z(z)
        edge_features = torch.cat([edge_length_embedded, scalar_z[edge_src], scalar_z[edge_dst]], dim=1)

        for i, lay in enumerate(self.layers):
            x = lay(
                x,
                z,
                edge_src,
                edge_dst,
                edge_sh,
                edge_features,
                edge_weight_cutoff,
            )

        x = self.hidden(x, z)
        x = self.output_layer(x, z)
        # x = x + pos

        # if t is not None:
        #      x = x * (1 + scale) + shift
        # self.layers[-1]

        if self.reduce_output:
            return scatter(x, batch, dim=0, dim_size=int(batch.max()) + 1).div(self.num_nodes**0.5)
        else:
            return x


class e3nn_md_module(L.LightningModule):

    def __init__(
        self,
        topology: Dict[str, torch.Tensor],
        init_lr=1e-4,
        **model_kwargs: Dict,
    ) -> None:
        """_summary_

        Parameters
        ----------
        init_lr : float, optional
            initial learning rate, by default 0.0001
        """
        super().__init__()
        self.save_hyperparameters()
        self.topology = topology

        # self.weight_fill = weight_fill
        self.model = Network(**model_kwargs)

        self.init_lr = init_lr

    #     if init_weight:
    #         self._init_model()

    # @torch.no_grad()
    # def _init_model(self):
    #     for _, param in self.named_parameters():
    #         param.data.fill_(self.weight_fill)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        # topology: Dict[str, torch.Tensor],
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """evaluate the network

        Parameters:
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``y`` the target, optional
            - ``batch`` the graph to which the node belong, optional
        """
        results = self.model(data, self.topology, pos)
        return results

    def _get_loss(self, data):

        prediction = self(data)
        return F.mse_loss(data["y"], prediction)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=0.01, lr=self.init_lr)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        )
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "validation_loss",
        }

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient of {name}: {param.grad}")
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
