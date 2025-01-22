import torch
import yaml


def tensor_to_dict(pose: torch.Tensor, node_attr=None):
    assert pose.shape[-1] == 3
    data = {"pos": pose}
    if node_attr:
        data["z"] = node_attr
    # data['pos'] = pose
    return data


def count_parameters(model, only_trainable=True):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and only_trainable:
            continue
        params = parameter.numel()
        print(name, params)
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def dict_from_yaml(yml_file):
    return yaml.safe_load(open(yml_file, "r"))


def dict_to_yaml(dict_t, yml_file):
    with open(yml_file, "w") as fp:
        yaml.dump(dict_t, fp, default_flow_style=False)
