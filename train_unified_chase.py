# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models_unified import create_model, MODEL_REGISTRY
from model import UNet as UNet_REF 

SEED = 42
DESIRED_SIZE = 1008
TRAIN_CFG = dict(
    input_channels=1,
    start_neurons=16,
    keep_prob=0.87,
    block_size=7,
    epochs=70,
    batch_size=1,
    lr=5e-4,
    weight_decay=1e-4,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def chase_paths(base_dir):
    base = Path(base_dir)
    paths = {
        "train_img":  base / "Chase" / "train"    / "image",
        "train_lbl":  base / "Chase" / "train"    / "label",
        "val_img":    base / "Chase" / "validate" / "images",
        "val_lbl":    base / "Chase" / "validate" / "labels",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"[{k}] 路径不存在: {p}")
    return paths

def match_label_name(image_path: Path) -> str:
    stem = image_path.stem
    return f"{stem}_1stHO.png"

def pad_to_desired(img, desired=DESIRED_SIZE):
    h, w = img.shape[:2]
    delta_w = desired - w
    delta_h = desired - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(img, (desired, desired))

def read_pair_full(img_path: Path, lbl_dir: Path):
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    lbl_name = match_label_name(img_path)
    lab = cv2.imread(str(lbl_dir / lbl_name), cv2.IMREAD_GRAYSCALE)
    if im is None or lab is None:
        raise RuntimeError(f"读取失败: {img_path} 或 {lbl_dir/lbl_name}")
    im = pad_to_desired(im, DESIRED_SIZE)
    lab = pad_to_desired(lab, DESIRED_SIZE)
    _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
    return (im.astype(np.float32)/255.0)[None,...], (lab.astype(np.float32)/255.0)[None,...]

class CHASEFullImage(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.imgs = sorted([p for p in Path(img_dir).iterdir()
                            if p.suffix.lower() in [".png", ".jpg", ".bmp", ".tif", ".tiff"]])
        self.lbl_dir = Path(lbl_dir)
        if not self.imgs:
            raise RuntimeError(f"未在 {img_dir} 发现图像文件")
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        x, y = read_pair_full(self.imgs[i], self.lbl_dir)
        return torch.from_numpy(x), torch.from_numpy(y)

def _resize_pos_embed_tensor(pe: torch.Tensor, target_tokens: int) -> torch.Tensor:
    import math
    B, N, C = pe.shape
    h_src = int(math.sqrt(N))
    if h_src * h_src != N:
        return pe
    grid = pe.reshape(1, h_src, h_src, C).permute(0,3,1,2)
    h_tgt = int(math.sqrt(target_tokens)); w_tgt = target_tokens // h_tgt
    grid = torch.nn.functional.interpolate(grid, size=(h_tgt, w_tgt), mode='bicubic', align_corners=False)
    grid = grid.permute(0,2,3,1).reshape(1, h_tgt*w_tgt, C)
    return torch.nn.Parameter(grid.to(pe.device), requires_grad=True)

def adapt_pos_embed_on_the_fly(model: nn.Module):
    for m in model.modules():
        if hasattr(m, 'pos_embed') and hasattr(m, 'patch_embed'):
            patch = getattr(m, 'patch_embed')
            if isinstance(patch, nn.Module):
                def make_hook(owner):
                    def hook(mod, inp, out):
                        if isinstance(out, torch.Tensor):
                            H, W = int(out.shape[-2]), int(out.shape[-1])
                            target_tokens = H * W
                            if owner.pos_embed.shape[1] != target_tokens:
                                owner.pos_embed = _resize_pos_embed_tensor(owner.pos_embed, target_tokens)
                    return hook
                patch.register_forward_hook(make_hook(m))

def unify_patch_size_runtime(model: nn.Module, target_ps: int = 16, in_ch: int = TRAIN_CFG["input_channels"]):

    replaced = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
            s = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
            if mod.in_channels == in_ch and k == s and (k != target_ps or s != target_ps):

                if ('patch' in name) or ('embed' in name):
                    parent = model
                    parts = name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    new_conv = nn.Conv2d(
                        in_channels=mod.in_channels,
                        out_channels=mod.out_channels,
                        kernel_size=target_ps, stride=target_ps,
                        padding=0, bias=(mod.bias is not None)
                    )
                    nn.init.kaiming_normal_(new_conv.weight, nonlinearity='relu')
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)
                    setattr(parent, parts[-1], new_conv)
                    replaced.append(name)
    if replaced:
        print(f"[unify_patch_size_runtime] 已将以下层的 patch_size 统一为 {target_ps}：{replaced}")
    else:
        print("[unify_patch_size_runtime] 未发现需要统一的 patch 层（可能已是 16 或写法不同）。")

def forward_model(model_name, model, x):
    if model_name.lower() == "medsegdiff":
        t = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return model(x, t)
    return model(x)

def build_model(model_name, device, target_ps=16):
    model = create_model(model_name,
                         input_channels=TRAIN_CFG["input_channels"],
                         start_neurons=TRAIN_CFG["start_neurons"]).to(device)

    unify_patch_size_runtime(model, target_ps=target_ps, in_ch=TRAIN_CFG["input_channels"])

    adapt_pos_embed_on_the_fly(model)
    return model

def build_loss(model_name):
    if model_name.lower() == "segmamba":
        return nn.BCEWithLogitsLoss()
    return nn.BCELoss()

def train_one_model(args, device, model_name, paths):
    ds_tr = CHASEFullImage(paths["train_img"], paths["train_lbl"])
    ds_va = CHASEFullImage(paths["val_img"],   paths["val_lbl"])
    dl_tr = DataLoader(ds_tr, batch_size=TRAIN_CFG["batch_size"], shuffle=True,
                       num_workers=args.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=TRAIN_CFG["batch_size"], shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = build_model(model_name, device, target_ps=args.force_patch_size)
    criterion = build_loss(model_name)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])

    use_amp = (args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.save_dir) / "Chase" / "checkpoint_unified_models" / model_name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{model_name.lower()}_best.pth"
    last_path = out_dir / f"{model_name.lower()}_last.pth"

    best_val = float("inf")
    for epoch in range(1, TRAIN_CFG["epochs"]+1):
        model.train(); tr_loss = 0.0
        for x,y in dl_tr:
            x=x.to(device); y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z = forward_model(model_name, model, x)
                loss = criterion(z,y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tr_loss += float(loss.detach().cpu().item())
        tr_loss /= max(1,len(dl_tr))

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x,y in dl_va:
                x=x.to(device); y=y.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    z = forward_model(model_name, model, x)
                    va_loss += criterion(z,y).item()
        va_loss /= max(1,len(dl_va))

        torch.save(model.state_dict(), last_path)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
        print(f"[{model_name.upper():10}] Epoch {epoch:03d}/{TRAIN_CFG['epochs']} | train {tr_loss:.4f} | val {va_loss:.4f} | best {best_val:.4f}")

    return str(best_path)

def main():
    parser = argparse.ArgumentParser(description="统一训练（CHASE_DB1 | 整图 | 4模型）")
    parser.add_argument("--data_dir", type=str, default=".", help="数据根目录（包含 Chase/ 子目录）")
    parser.add_argument("--save_dir", type=str, default=".", help="权重输出根目录")
    parser.add_argument("--models", type=str, default="transunet,swinunet,medsegdiff,segmamba",
                        help="逗号分隔：transunet,swinunet,medsegdiff,segmamba")
    parser.add_argument("--force_patch_size", type=int, default=16, help="运行时强制将 patch-embed 改为该 patch_size（建议16）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers，Windows 建议 0")
    parser.add_argument("--amp", action="store_true", default=True, help="启用混合精度(默认开)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(args.device)
    paths = chase_paths(args.data_dir)

    all_models = list(MODEL_REGISTRY.keys())
    target = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    bad = [m for m in target if m not in all_models or m=='unet']
    if bad:
        print(f"⚠️ 忽略/非法模型: {bad}（本脚本不训练 UNet）")
        target = [m for m in target if m in all_models and m!='unet']

    bests = {}
    for name in target:
        bests[name] = train_one_model(args, device, name, paths)

    print("\\n=== 最佳权重清单 ===")
    for k,v in bests.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()