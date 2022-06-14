import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
import cv2

def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / std
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered

def to255(feat, mmin=None, mmax=None):
    if mmin is None:
        mmax = np.max(feat)
        mmin = np.min(feat)
    # mmax, mmin = 10, -10
    k = (255 - 0) / (mmax - mmin)
    normed = 0 + k * (feat - mmin)
    return np.clip(normed.astype(int), 0, 255)
    # return torch.clamp(normed.int(), 0, 255).cpu().numpy()


def convert_overlay_heatmap(feat_map, img, alpha = 0.5, mmin=None, mmax=None):
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    if mmax is None:
        norm_img = np.zeros(feat_map.shape)
        norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
        # print(norm_img)
        # print(feat_map.min(), feat_map.max())
    else:
        norm_img = to255(feat_map, mmin, mmax)
        # print(norm_img)
    print(norm_img.max())
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


bins = np.arange(-2., 2.1, 0.1)
# bins[0] = -10
# bins[-1] = 10
y = [list(np.arange(0, 9e5, 1e5)), list(np.arange(0, 3.1e5, 3e4)),
     list(np.arange(0, 9e4, 1e4)), list(np.arange(0, 3.1e4, 3e3))]
for i, out in enumerate(outs[:-1]):
    score = pd.cut(out.reshape(-1), bins)
    res = pd.value_counts(score)
    res = list(res.sort_index())
    plt.bar(list(np.arange(-2., 2., 0.1)), res)
    plt.yticks(y[i], fontsize=12)
    plt.xlabel('feature map ranges', fontsize=16)
    plt.ylabel('number of elements', fontsize=16)
    plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    # plt.show()
    plt.rcParams['savefig.dpi'] = 496
    plt.savefig(f'gfl2/gfl_r101_fpn{i}_dis.png', bbox_inches='tight',
                pad_inches=0)
    plt.cla()

# print(outs[-1].shape)
# torch.save(norm(outs[-2]), 'fcos_x101_fpn3.pth')
# assert False

# size = outs[0].shape[2:]
# fusions = []
# C = 256
# for i, out in enumerate(outs[:-1]):
#     fusion = F.interpolate(out, size, mode='bilinear')
#     fusion = fusion.permute(1, 0, 2, 3).reshape(C, -1)
#     fusions.append(fusion)
# fusions = torch.cat(fusions, dim=-1)
# mean = fusions.mean(dim=-1, keepdim=True)
# mean = mean.unsqueeze(0).unsqueeze(-1)
# std = fusions.std(dim=-1, keepdim=True)
# std = std.unsqueeze(0).unsqueeze(-1)
# for i, out in enumerate(outs[:-1]):
#     img = cv2.imread('demo/demo.jpg')
#     img = img[..., ::-1]
#     size = img.shape[:2]
#
#     out = F.interpolate(out, size, mode='bilinear')
#     out = (out - mean) / std
#     # out = torch.abs(out)
#     act_max = torch.max(out, dim=1)[0]
#     act_max = convert_overlay_heatmap(act_max[0], img)
#     plt.axis('off')
#     plt.imshow(act_max, cmap='Reds')
#     plt.show()
# assert False

mmin_1, mmax_1 = 100, -100
mmin_2, mmax_2 = 100, -100
img = cv2.imread('demo/demo.jpg')
size = img.shape[:2]
for i, out in enumerate(outs[:-1]):
    out = F.interpolate(out, size, mode='bilinear')
    act_max = torch.max(out, dim=1)[0]
    mmin_1 = min(mmin_1, act_max.min())
    mmax_1 = max(mmax_1, act_max.max())
    out = norm(out)
    act_max = torch.max(out, dim=1)[0]
    mmin_2 = min(mmin_2, act_max.min())
    mmax_2 = max(mmax_2, act_max.max())
print(mmin_1, mmax_1)
print(mmin_2, mmax_2)

for i, out in enumerate(outs[:-1]):
    # x = list(range(256))
    # y1 = [0] * 256
    # act_max = torch.max(out, dim=1)[1]
    # for a in act_max[0]:
    #     for b in a:
    #         y1[b] += 1
    # y2 = [0] * 256
    # out = norm(out)
    # act_max = torch.max(out, dim=1)[1]
    # for a in act_max[0]:
    #     for b in a:
    #         y2[b] += 1
    # l1 = plt.plot(x, y1, label='vanilla')
    # l2 = plt.plot(x, y2, label='normalized')
    # plt.xlabel('channel index', fontsize=18)
    # plt.ylabel('number', fontsize=18)
    # # plt.xticks(fontsize=14)
    # # plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    # plt.show()
    # break

    # dic = {x: 0 for x in range(256)}
    # act_max = torch.max(out, dim=1)[1]
    # for a in act_max[0]:
    #     for b in a:
    #         dic[int(b)] += 1
    # dic = sorted(dic.items(), key=lambda  x:x[1], reverse=True)
    # print(dic)
    #
    # dic = {x: 0 for x in range(256)}
    # out = norm(out)
    # act_max = torch.max(out, dim=1)[1]
    # for a in act_max[0]:
    #     for b in a:
    #         dic[int(b)] += 1
    # dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    # print(dic)
    # break

    # out = torch.clamp(out, mmin, mmax)
    # print('mean: ', out.mean(), 'std: ', out.std(), 'norm: ', torch.norm(out, 2),
    #       'max: ', out.max(), 'min: ', out.min())
    # print(out.shape)
    # out = out[:, 0:1]
    # act_max = torch.max(out, dim=1)[0]
    # act_max = to255(act_max)[0]
    # plt.axis('off')
    # plt.imshow(act_max, cmap='Reds')
    # plt.show()
    # out = out_copy[:, 0:1]
    # out = norm(out)
    # out = torch.clamp(out, mmin, mmax)
    # print('mean: ', out.mean(), 'std: ', out.std(), 'norm: ', torch.norm(out, 2),
    #       'max: ', out.max(), 'min: ', out.min())
    # print(pd.value_counts(pd.cut(out.reshape(-1), bins)))
    # break
    if i >= 0:
        #
        # out = F.interpolate(out, size, mode='bilinear')
        # act_max = torch.max(out, dim=1)[0]
        # act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6)
        # plt.axis('off')
        # plt.imshow(act_max, cmap='Reds')
        # plt.show()
        #
        out_copy = copy.deepcopy(out)
        out = F.interpolate(out_copy, size, mode='bilinear')
        print(out.min(), out.max())
        act_max = torch.max(out, dim=1)[0]
        # act_max = out[0, 89].unsqueeze(0)
        act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6,
                                          mmin=mmin_1.item(),
                                          mmax=mmax_1.item())
        # act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6,
        #                                   mmin=act_max.min().item(), mmax=act_max.max().item())
        plt.axis('off')
        plt.imshow(act_max, cmap='Reds')
        # plt.show()
        plt.savefig(f'gfl2/gfl_r101_fpn{i}.png', bbox_inches='tight',
                    pad_inches=0)

        # out = F.interpolate(out_copy, size, mode='bilinear')
        # act_max = torch.max(out, dim=1)[0]
        # # act_max = out[0, 89].unsqueeze(0)
        # act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6)
        # plt.axis('off')
        # plt.imshow(act_max, cmap='Reds')
        # plt.show()

        # out = F.interpolate(out_copy, size, mode='bilinear')
        # out = norm(out)
        # print(out.min(), out.max())
        # act_max = torch.max(out, dim=1)[0]
        # # act_max = out[0, 147].unsqueeze(0)
        # act_max = convert_overlay_heatmap(act_max[0], img, alpha=0.6,
        #                                   mmin=mmin_2.item(), mmax=mmax_2.item())
        # plt.axis('off')
        # plt.imshow(act_max, cmap='Reds')
        # # plt.show()
        # plt.savefig(f'gfl/frcnn_fpn{i}_norm.png', bbox_inches='tight',
        #             pad_inches=0)