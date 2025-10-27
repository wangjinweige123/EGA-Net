
import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

import torch

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models_unified import create_model, MODEL_REGISTRY
from model import UNet as UNet_REF

DEFAULTS = dict(
    threshold=0.5,
    desired_size=1008,
    start_neurons=16,
    input_channels=1,
    warmup=20,
    runs=200,
)

REFERENCE_SIZE = 512

MODEL_EVAL_SIZE = {
    'unet': 1008,
    'transunet': 512,
    'swinunet': 512,
    'medsegdiff': 384,
    'segmamba': 512,
    'attentionunet': 1008,
    'unetplusplus': 1008,
    'vlight': 1008,
    'deeplabv3plus': 1008,
    'mobilenetv3_lraspp': 1008,
}

import torch.nn as nn
def _resize_pos_embed_tensor(pe: torch.Tensor, target_tokens: int) -> torch.Tensor:
    import math
    B, N, C = pe.shape
    h_src = int(math.sqrt(N))
    if h_src * h_src != N:
        return pe
    grid = pe.reshape(1, h_src, h_src, C).permute(0,3,1,2)
    h_tgt = int(math.sqrt(target_tokens)); w_tgt = target_tokens // h_tgt
    grid = torch.nn.functional.interpolate(grid, size=(h_tgt, w_tgt),
                                           mode='bicubic', align_corners=False)
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

def unify_patch_size_runtime(model: nn.Module, target_ps: int = 16, in_ch: int = DEFAULTS["input_channels"]):
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
                    try:
                        new_conv = new_conv.to(mod.weight.device)
                    except Exception:
                        pass
                    nn.init.kaiming_normal_(new_conv.weight, nonlinearity='relu')
                    if new_conv.bias is not None:
                        nn.init.zeros_(new_conv.bias)
                    setattr(parent, parts[-1], new_conv)
                    replaced.append(name)
    if replaced:
        print(f"[unify_patch_size_runtime] 已将以下层的 patch_size 统一为 {target_ps}:{replaced}")

def infer_start_neurons_from_sd(model_name: str, sd: dict, default: int = DEFAULTS["start_neurons"]) -> int:
    try:
        model_name_lower = model_name.lower()
        
        if model_name_lower == 'transunet':
            if 'conv1.0.weight' in sd:
                channels = int(sd['conv1.0.weight'].shape[0])
                print(f"[INFO] 从 conv1.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'swinunet':
            for key in ['input_stem.0.weight', 'patch_embed.weight', 'decoder1.0.weight']:
                if key in sd:
                    channels = int(sd[key].shape[0])
                    print(f"[INFO] 从 {key} 推断 {model_name} 的 start_neurons={channels}")
                    return channels
        
        elif model_name_lower == 'segmamba':
            if 'stem.0.weight' in sd:
                channels = int(sd['stem.0.weight'].shape[0])
                print(f"[INFO] 从 stem.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'medsegdiff':
            if 'input_blocks.0.0.weight' in sd:
                channels = int(sd['input_blocks.0.0.weight'].shape[0])
                print(f"[INFO] 从 input_blocks.0.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
            elif 'model.input_blocks.0.0.weight' in sd:
                channels = int(sd['model.input_blocks.0.0.weight'].shape[0])
                print(f"[INFO] 从 model.input_blocks.0.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'unet':
            if 'enc1.0.weight' in sd:
                channels = int(sd['enc1.0.weight'].shape[0])
                print(f"[INFO] 从 enc1.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'attentionunet':
            if 'enc1.0.weight' in sd:
                channels = int(sd['enc1.0.weight'].shape[0])
                print(f"[INFO] 从 enc1.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'unetplusplus':
            if 'conv0_0.conv1.weight' in sd:
                channels = int(sd['conv0_0.conv1.weight'].shape[0])
                print(f"[INFO] 从 conv0_0.conv1.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'vlight':
            if 'enc1.0.conv.weight' in sd:
                channels = int(sd['enc1.0.conv.weight'].shape[0])
                print(f"[INFO] 从 enc1.0.conv.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'deeplabv3plus':
            if 'enc1.0.weight' in sd:
                channels = int(sd['enc1.0.weight'].shape[0])
                print(f"[INFO] 从 enc1.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        elif model_name_lower == 'mobilenetv3_lraspp':
            if 'conv1.0.weight' in sd:
                channels = int(sd['conv1.0.weight'].shape[0])
                print(f"[INFO] 从 conv1.0.weight 推断 {model_name} 的 start_neurons={channels}")
                return channels
        
        for key in sorted(sd.keys()):
            if 'weight' in key and isinstance(sd[key], torch.Tensor):
                shape = sd[key].shape
                if len(shape) == 4 and shape[1] == 1:
                    channels = int(shape[0])
                    print(f"[INFO] 从 {key} 推断 {model_name} 的 start_neurons={channels}")
                    return channels
        
    except Exception as e:
        print(f"[WARNING] 推断 start_neurons 失败: {e}")
    
    print(f"[INFO] 使用默认 start_neurons={default}")
    return int(default)

def filter_state_dict_by_shape(model: torch.nn.Module, sd: dict) -> dict:
    msd = model.state_dict()
    new_sd = {}
    for k, v in sd.items():
        if k in msd and isinstance(v, torch.Tensor) and tuple(v.shape) == tuple(msd[k].shape):
            new_sd[k] = v
    missing = [k for k in msd.keys() if k not in new_sd]
    if missing:
        print(f"[eval] 将跳过 {len(missing)} 个形状不匹配或缺失的权重键(例如: {missing[:3]} ...)")
    return new_sd

def align_model_pos_embed_to_sd(model: torch.nn.Module, sd: dict):
    try:
        for key in ['pos_embed', 'module.pos_embed']:
            if hasattr(model, 'pos_embed') and key in sd and isinstance(sd[key], torch.Tensor):
                m_pe = model.pos_embed
                if isinstance(m_pe, torch.nn.Parameter) and int(m_pe.shape[1]) != int(sd[key].shape[1]):
                    device = m_pe.device
                    with torch.no_grad():
                        new_param = torch.nn.Parameter(sd[key].to(device).clone(), requires_grad=m_pe.requires_grad)
                        model.pos_embed = new_param
                    print(f"[eval] 已将模型 pos_embed 参数形状调整为 checkpoint: {tuple(sd[key].shape)}")
                    break
    except Exception as e:
        print(f"[eval] 调整模型 pos_embed 形状失败: {e}")

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

def list_chase_test_paths(base_dir):
    base = Path(base_dir)
    paths = {
        "test_img":  base / "Chase" / "test" / "image",
        "test_lbl":  base / "Chase" / "test" / "label",
        "orig_img":  base / "Chase" / "test" / "im",  # 添加原始彩色图像路径
    }
    for k, p in paths.items():
        if not p.exists() and k != "orig_img":  # 原始图像路径可选
            raise FileNotFoundError(f"[{k}] 路径不存在: {p}")
    return paths

def match_label_name(image_path: Path) -> str:
    stem = image_path.stem
    return f"{stem}_1stHO.png"

def pad_to_desired(img, desired):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("输入图像尺寸非法:h or w == 0")

    scale = min(desired / float(h), desired / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    if len(img.shape) == 2:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, desired - new_h)
    pad_w = max(0, desired - new_w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def read_pair_full(img_path: Path, lbl_dir: Path, desired):
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    lbl_name = match_label_name(img_path)
    lab = cv2.imread(str(lbl_dir / lbl_name), cv2.IMREAD_GRAYSCALE)
    if im is None or lab is None:
        raise RuntimeError(f"读取失败: {img_path} 或 {lbl_dir/lbl_name}")
    
    im = pad_to_desired(im, desired)
    
    h, w = lab.shape[:2]
    scale = min(desired / float(h), desired / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    lab_resized = cv2.resize(lab, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_h = max(0, desired - new_h)
    pad_w = max(0, desired - new_w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    lab_padded = cv2.copyMakeBorder(lab_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    _, lab = cv2.threshold(lab_padded, 127, 255, cv2.THRESH_BINARY)
    
    return (im.astype(np.float32)/255.0), (lab.astype(np.float32)/255.0)

def sigmoid_np(x): 
    return 1.0 / (1.0 + np.exp(-x))

def forward_model(model_name, model, x):
    if model_name.lower() == "medsegdiff":
        t = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return model(x, t)
    return model(x)

def predict_full_image(model_name, model, image, device):
    x = torch.from_numpy(image[None, None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        out = forward_model(model_name, model, x)
        out_np = out.detach().float().cpu().numpy()[0,0]
        if (out_np.max() > 1.0) or (out_np.min() < 0.0):
            out_np = sigmoid_np(out_np)
        return out_np

def confusion_from_binary(y_true, y_pred_bin):
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    return tp, tn, fp, fn

def metrics_from_scores(y_true, y_prob, thr=0.5):
    y_true = y_true.astype(np.uint8).ravel()
    y_prob = y_prob.astype(np.float32).ravel()
    y_bin  = (y_prob >= thr).astype(np.uint8)
    tp, tn, fp, fn = confusion_from_binary(y_true, y_bin)
    sen = tp / max(1, (tp + fn))
    spe = tn / max(1, (tn + fp))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    f1  = (2*tp) / max(1, (2*tp + fp + fn))
    if SKLEARN_OK:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = float("nan")
    else:
        order = np.argsort(y_prob)
        y_sorted = y_true[order]
        tps = np.cumsum(y_sorted[::-1] == 1)
        fps = np.cumsum(y_sorted[::-1] == 0)
        tpr = tps / max(1, np.sum(y_true == 1))
        fpr = fps / max(1, np.sum(y_true == 0))
        auc = float(np.trapz(tpr, fpr))
    return dict(F1=f1, AUC=auc, Sen=sen, Spe=spe, Acc=acc)

def search_best_threshold(y_true_list, y_prob_list, grid=None):
    import numpy as np
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    y_true = np.concatenate([y.ravel().astype(np.uint8) for y in y_true_list])
    y_prob = np.concatenate([p.ravel().astype(np.float32) for p in y_prob_list])
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        y_bin = (y_prob >= t).astype(np.uint8)
        tp = np.sum((y_true == 1) & (y_bin == 1))
        fp = np.sum((y_true == 0) & (y_bin == 1))
        fn = np.sum((y_true == 1) & (y_bin == 0))
        f1 = (2*tp) / max(1, (2*tp + fp + fn))
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_fps(model_name, model, device, size=1008, warmup=20, runs=200):
    model.eval()
    x = torch.randn(1, 1, size, size, device=device)
    
    if model_name.lower() == 'medsegdiff':
        t = torch.zeros(1, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(warmup):
            if model_name.lower() == 'medsegdiff':
                _ = model(x, t)
            else:
                _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(0.1)
    
    timings = []
    
    with torch.no_grad():
        for _ in range(runs):
            if device.type == 'cuda':
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
            else:
                start_time = time.time()
            
            if model_name.lower() == 'medsegdiff':
                _ = model(x, t)
            else:
                _ = model(x)
            
            if device.type == 'cuda':
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
            else:
                end_time = time.time()
                timings.append((end_time - start_time) * 1000)
    
    timings = sorted(timings)
    trim_count = max(1, int(runs * 0.05))
    valid_timings = timings[trim_count:-trim_count]
    time_ms = float(np.mean(valid_timings))
    fps = 1000.0 / time_ms if time_ms > 0 else float('inf')
    
    return fps, time_ms

def estimate_flops(model, device, size=1008):
    import torch.nn as nn
    def is_supported(m):
        return isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                              nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d))
    def add_reset(m):
        if is_supported(m): m.__flops__ = 0
    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            batch = inp[0].shape[0]
            out_c, out_h, out_w = out.shape[1:]
            k_h, k_w = m.kernel_size
            in_c = m.in_channels
            flops = batch * out_h * out_w * (k_h * k_w * in_c) * out_c
            if m.bias is not None: flops += batch * out_h * out_w * out_c
            m.__flops__ += int(flops)
        elif isinstance(m, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d)):
            batch = inp[0].shape[0]
            elements = batch * int(np.prod(out.shape[1:]))
            m.__flops__ += elements
        elif isinstance(m, nn.ConvTranspose2d):
            batch = inp[0].shape[0]
            out_h, out_w = out.shape[2:]
            k_h, k_w = m.kernel_size
            in_c = m.in_channels
            flops = batch * out_h * out_w * (k_h * k_w * in_c)
            if m.bias is not None:
                flops += batch * out_h * out_w * m.out_channels
            m.__flops__ += int(flops)
    model.apply(add_reset)
    handles = []
    for m in model.modules():
        if is_supported(m):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    x = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        _ = model(x)
    for h in handles: h.remove()
    total = 0
    for m in model.modules():
        if hasattr(m, "__flops__"):
            total += m.__flops__
    return total


def get_original_image(image_path: Path, orig_img_dir: Path, default_gray_img: np.ndarray):

    stem = image_path.stem  
    base_name = f"{stem}"

    orig_img_path = orig_img_dir / f"{base_name}.png"
    
    if not orig_img_path.exists():
        for ext in ['.jpg', '.JPG', '.png', '.PNG']:
            alt_path = orig_img_dir / f"{base_name}{ext}"
            if alt_path.exists():
                orig_img_path = alt_path
                break
    
    if orig_img_path.exists():
        orig_img = cv2.imread(str(orig_img_path))
        if orig_img is not None:
            print(f"读取原始图像: {orig_img_path}")
            return orig_img

    print(f"未找到原始图像: {orig_img_path}，使用灰度图像代替")
    if len(default_gray_img.shape) == 2:

        gray_uint8 = (default_gray_img * 255).astype(np.uint8)
        orig_img = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    else:
        orig_img = default_gray_img
    
    return orig_img

def create_binary_image(mask):
    h, w = mask.shape
    binary_img = np.zeros((h, w, 3), dtype=np.uint8)
    binary_img[mask > 0.5] = [255, 255, 255]
    return binary_img


def create_overlay_image(gray_img, gt_mask, pred_mask):
    gray_uint8 = (gray_img * 255).astype(np.uint8)
    overlay_img = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

    overlay_img[gt_mask > 0.5, 1] = 255

    overlay_img[pred_mask > 0.5, 2] = 255
    
    return overlay_img
def combine_visualizations(orig_img, gt_img, pred_img, overlay_img):

    h, w = orig_img.shape[:2]
    title_h = 30
    canvas = np.ones((h + title_h, w * 4, 3), dtype=np.uint8) * 255

    titles = [
        ('Original Image', w//2 - 80),
        ('Ground Truth', w + w//2 - 80),
        ('Prediction', 2*w + w//2 - 60),
        ('Overlay (GT-green / Pred-red)', 3*w + w//2 - 140)
    ]
    
    for title, pos_x in titles:
        cv2.putText(canvas, title, (pos_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    canvas[title_h:, 0:w] = orig_img
    canvas[title_h:, w:2*w] = gt_img
    canvas[title_h:, 2*w:3*w] = pred_img
    canvas[title_h:, 3*w:4*w] = overlay_img
    
    return canvas


def save_visualizations(model_name, img_files, images, y_true_all, y_prob_all, 
                       threshold, paths, eval_size):
    orig_path = Path(paths.get("orig_img", "."))
    vis_dir = orig_path.parent / "visualization" / f"visualization-{model_name}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n生成可视化结果到: {vis_dir}")
    
    for i, (img_file, gray_img, gt_mask, pred_prob) in enumerate(zip(img_files, images, y_true_all, y_prob_all)):

        pred_mask = (pred_prob > threshold).astype(np.float32)

        orig_img_dir = Path(paths.get("orig_img", "."))
        orig_img = get_original_image(img_file, orig_img_dir, gray_img)

        h, w = pred_prob.shape[:2]
        orig_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_AREA)

        gt_img = create_binary_image(gt_mask)
        pred_img = create_binary_image(pred_mask)
        overlay_img = create_overlay_image(gray_img, gt_mask, pred_mask)

        canvas = combine_visualizations(orig_img, gt_img, pred_img, overlay_img)

        output_path = vis_dir / f"{i}_{img_file.stem}_comparison.png"
        cv2.imwrite(str(output_path), canvas)
        print(f"已保存可视化结果: {output_path}")

def save_raw_predictions(model_name, img_files, y_prob_all, threshold, paths):

    base = Path(paths["test_img"]).parent 
    res_dir = base / "result" / f"result-eval_{model_name}"
    res_dir.mkdir(parents=True, exist_ok=True)

    for i, (fp, prob) in enumerate(zip(img_files, y_prob_all)):
        prob_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        bin_u8  = ((prob >= threshold).astype(np.uint8) * 255)
        cv2.imwrite(str(res_dir / f"{i}_{fp.stem}.png"), prob_u8)      
        cv2.imwrite(str(res_dir / f"{i}_{fp.stem}_bin.png"), bin_u8)   


def main():
    parser = argparse.ArgumentParser(description="统一评估(CHASE_DB1 | 整图 | 10模型)")
    parser.add_argument("--force_eval_size", type=int, default=0,
                    help=">0 时强制所有模型使用该分辨率做整图评估")
    parser.add_argument("--data_dir", type=str, default=".", help="数据根目录(包含 Chase/ 子目录)")
    parser.add_argument("--weights_dir", type=str, default=".", help="权重根目录")
    parser.add_argument("--models", type=str, default="all",
                        help="逗号分隔模型名或 all")
    parser.add_argument("--unet_weight_path", type=str, default="Chase/test/checkpoint/UNet_base.pth")
    parser.add_argument("--transunet_weight_path", type=str, default="")
    parser.add_argument("--swinunet_weight_path", type=str, default="")
    parser.add_argument("--medsegdiff_weight_path", type=str, default="")
    parser.add_argument("--segmamba_weight_path", type=str, default="")
    parser.add_argument("--attentionunet_weight_path", type=str, default="")
    parser.add_argument("--unetplusplus_weight_path", type=str, default="")
    parser.add_argument("--vlight_weight_path", type=str, default="")
    parser.add_argument("--deeplabv3plus_weight_path", type=str, default="")
    parser.add_argument("--mobilenetv3_lraspp_weight_path", type=str, default="")
    
    parser.add_argument("--force_patch_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=DEFAULTS["threshold"])
    parser.add_argument("--auto_thr", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=int, default=DEFAULTS["runs"])
    parser.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    parser.add_argument("--save_csv", type=str, default="results_unified_chase.csv")
    parser.add_argument("--save_vis", type=int, default=1, help="是否保存可视化结果(1=是,0=否)")
    
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    paths = list_chase_test_paths(args.data_dir)

    all_models = list(MODEL_REGISTRY.keys())
    target = all_models if args.models.strip().lower()=="all" else [m.strip().lower() for m in args.models.split(",")]
    bad = [m for m in target if m not in all_models]
    if bad:
        raise ValueError(f"未知模型: {bad} | 可选: {all_models}")

    img_files = sorted([f for f in Path(paths["test_img"]).iterdir()
                        if f.suffix.lower() in [".png",".jpg",".bmp",".tif",".tiff"]])

    rows = []
    for model_name in target:
        print(f"\n===== 评估模型: {model_name.upper()}(整图) =====")

        eval_size = MODEL_EVAL_SIZE.get(model_name, DEFAULTS["desired_size"])
        if args.force_eval_size and args.force_eval_size > 0:
            eval_size = int(args.force_eval_size)

        print(f"[INFO] 使用评估分辨率: {eval_size}×{eval_size}")

        images, labels = [], []
        for fp in img_files:
            im, lab = read_pair_full(fp, Path(paths["test_lbl"]), eval_size)
            images.append(im)
            labels.append(lab)

        if model_name == "unet":
            model = UNet_REF(input_channels=1, start_neurons=16, keep_prob=0.87, block_size=7,
                             use_saspp=False, use_ega=False).to(device)
            ckpt = Path(args.unet_weight_path)
            if not ckpt.exists():
                raise FileNotFoundError(f"未找到 UNet 旧权重: {ckpt}")
        else:
            explicit = {
                'transunet': args.transunet_weight_path,
                'swinunet': args.swinunet_weight_path,
                'medsegdiff': args.medsegdiff_weight_path,
                'segmamba': args.segmamba_weight_path,
                'attentionunet': args.attentionunet_weight_path,
                'unetplusplus': args.unetplusplus_weight_path,
                'vlight': args.vlight_weight_path,
                'deeplabv3plus': args.deeplabv3plus_weight_path,
                'mobilenetv3_lraspp': args.mobilenetv3_lraspp_weight_path,
            }.get(model_name, "")
            if explicit:
                ckpt = Path(explicit)
            else:
                base_unified = Path(args.weights_dir) / "Chase" / "checkpoint_unified_models" / model_name
                ckpt = base_unified / f"{model_name}_best.pth"
                if not ckpt.exists():
                    print(f"⚠️ 未找到权重: {ckpt},尝试 last.pth")
                    ckpt = base_unified / f"{model_name}_last.pth"
                if not ckpt.exists():
                    base_test = Path(args.weights_dir) / "Chase" / "test" / "checkpoint" / model_name
                    alt = base_test / f"{model_name}_best.pth"
                    if alt.exists():
                        print(f"[eval] 使用回退权重: {alt}")
                        ckpt = alt
                    else:
                        alt2 = base_test / f"{model_name}_last.pth"
                        if alt2.exists():
                            print(f"[eval] 使用回退权重: {alt2}")
                            ckpt = alt2
                        else:
                            print(f"❌ 未找到任何权重,跳过该模型")
                            continue
            
            print(f"[INFO] 正在加载权重: {ckpt}")
            sd_tmp = torch.load(str(ckpt), map_location=device)
            start_c = infer_start_neurons_from_sd(model_name, sd_tmp, DEFAULTS["start_neurons"])
            
            model = create_model(model_name, input_channels=1, start_neurons=start_c).to(device)
            if model_name in {"transunet"}:
                unify_patch_size_runtime(model, target_ps=args.force_patch_size)
            adapt_pos_embed_on_the_fly(model)

        sd = torch.load(str(ckpt), map_location=device)
        align_model_pos_embed_to_sd(model, sd)
        sd_filtered = filter_state_dict_by_shape(model, sd)
        msd = model.state_dict()
        matched = len([k for k in sd_filtered.keys() if k in msd])
        total = len([k for k in msd.keys()])
        cov = 100.0 * matched / max(1, total)
        print(f"[eval] 权重形状匹配覆盖率: {matched}/{total} ({cov:.1f}%)")
        
        model.load_state_dict(sd_filtered, strict=False)
        model.eval()

        fps, time_ms = measure_fps(model_name, model, device, size=eval_size,
                                   warmup=args.warmup, runs=args.runs)
        flops = estimate_flops(model, device, size=eval_size)
        params, _ = count_params(model)
        params_m = params / 1e6
        flops_g = flops / 1e9

        start_t = time.time()
        y_true_all, y_prob_all = [], []
        for im, lab in zip(images, labels):
            prob = predict_full_image(model_name, model, im, device)
            y_true_all.append(lab.astype(np.uint8))
            y_prob_all.append(prob.astype(np.float32))
        end_t = time.time()
        avg_infer_ms = ((end_t - start_t) / max(1, len(images))) * 1000.0

        if args.auto_thr == 1:
            best_thr, best_f1 = search_best_threshold(y_true_all, y_prob_all)
        else:
            best_thr, best_f1 = float(args.threshold), None
        print(f"[THR] 使用阈值: {best_thr:.2f}" + (f"（Auto, F1@thr={best_f1:.4f}）" if best_f1 is not None else ""))

        if args.save_vis == 1:
            save_visualizations(model_name, img_files, images, y_true_all, y_prob_all, 
                              best_thr, paths, eval_size)
        save_raw_predictions(model_name, img_files, y_prob_all, best_thr, paths)

        mets_per_image = [metrics_from_scores(l, p, thr=best_thr) for l, p in zip(y_true_all, y_prob_all)]
        mean_m = {k: float(np.mean([d[k] for d in mets_per_image])) for k in mets_per_image[0].keys()}

        normalized_flops_g = flops_g * (REFERENCE_SIZE / eval_size) ** 2

        row = dict(
            Model=model_name.upper(),
            # EvalSize=eval_size,
            Thr=round(float(best_thr), 2),
            BestF1=round(float(best_f1) if (best_f1 is not None) else float(mean_m["F1"]), 4),
            F1=round(mean_m["F1"], 4),
            AUC=round(mean_m["AUC"], 4),
            Sen=round(mean_m["Sen"], 4),
            Spe=round(mean_m["Spe"], 4),
            Acc=round(mean_m["Acc"], 4),
            Params_M=round(params_m, 2),
            FLOPs_G=round(flops_g, 2),
            FLOPs_G_512=round(normalized_flops_g, 2),
            FPS=round(float(fps), 2),
            Time_ms=round(float(time_ms), 2),
            Weights=str(ckpt)
        )

        print(f"[E2E] AvgInferTime_ms = {avg_infer_ms:.1f}")
        print(json.dumps(row, ensure_ascii=False, indent=2))
        rows.append(row)

    if rows:
        import csv
        rows.sort(key=lambda d: float(d.get("F1", 0.0)), reverse=True)
        out_csv = Path(args.save_csv)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n✅ 已保存评估结果表: {out_csv}")
    else:
        print("\n未生成评估结果。")


if __name__ == "__main__":
    main()