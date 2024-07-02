from yowo.models.backbone import build_backbone_2d, build_backbone_3d
from config.yowo_v2_config import yowo_v2_config

import torch

cfg = yowo_v2_config['yowo_v2_nano']

backbone_2d, dims_2d = build_backbone_2d(cfg, pretrained=False)
backbone_3d, dims_3d = build_backbone_3d(cfg, pretrained=False)

print("\n", dims_2d, dims_3d)

inp = torch.randn(1, 3, 16, 224, 224)

cls, reg = backbone_2d(inp[:, :, -1, ...].float())
print("2D out")
print(cls[0].shape)
print(cls[1].shape)
print(cls[2].shape)
print()
print(reg[0].shape)
print(reg[1].shape)
print(reg[2].shape)
print("\n3D out")
out = backbone_3d(inp.float())
print(out.shape)