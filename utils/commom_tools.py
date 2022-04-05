import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if "Normalize" in str(transform_train):
        norm_transform = list(
            filter(
                lambda x: isinstance(x, transforms.Normalize),
                transform_train.transforms,
            )
        )
        mean = torch.tensor(
            norm_transform[0].mean, dtype=img_.dtype, device=img_.device
        )
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype("uint8")).convert("RGB")
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype("uint8").squeeze())
    else:
        raise Exception(
            "Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(
                img_.shape[2]
            )
        )

    return img_


def get_attention_map(
    img, model, transform, cam_label=None, grid_shape=(31, 10), get_mask=False
):
    x = transform(img)
    _, att_mat = model(x.unsqueeze(0), cam_label=[5], get_attn_weights=True)
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    # print("att_mat: ", att_mat.shape)

    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # print("aug_att_mat: ", aug_att_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
    # print("joint_attentions: ", joint_attentions.shape)

    v = joint_attentions[-1]
    # print("v: ", v.shape)
    # print("grid_shape: ", grid_shape)

    mask = (
        v[0, 1 : 1 + grid_shape[0] * grid_shape[1]].reshape(grid_shape).detach().numpy()
    )
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

    return result


def get_template_attention_map(
    img, model, transform, cam_label=None, grid_shape=(31, 10), get_mask=False
):
    x = transform(img)
    _, att_mat = model(x.unsqueeze(0), cam_label=[5], get_attn_weights=True)
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    # print("att_mat: ", att_mat.shape)

    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # print("aug_att_mat: ", aug_att_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
    # print("joint_attentions: ", joint_attentions.shape)

    v = joint_attentions[-1]
    # print("v: ", v.shape)
    # print("grid_shape: ", grid_shape)

    mask = (
        v[0, 1 + grid_shape[0] * grid_shape[1] :].reshape(grid_shape).detach().numpy()
    )
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

    return result


def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax2.set_title("Attention Map Last Layer")
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
