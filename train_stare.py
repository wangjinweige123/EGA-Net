import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
from model import UNet
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size    = 1
epochs        = 70
learning_rate = 5e-4
weight_decay  = 1e-4
start_neurons = 16
keep_prob     = 0.87
block_size    = 7
desired_size  = 1008

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='UNet Training with/without SASPP and EGA - Stare Dataset')
parser.add_argument('--use_saspp', action='store_true', help='启用SASPP模块')
parser.add_argument('--use_ega', action='store_true', help='启用EGA模块')
parser.add_argument('--restore', action='store_true', help='从已有权重恢复训练')
args = parser.parse_args()
use_saspp = args.use_saspp
use_ega = args.use_ega

modules = []
if use_saspp:
    modules.append('saspp')
if use_ega:
    modules.append('ega')
    
module_name = '_'.join(modules) if modules else 'base'

module_info = []
if use_saspp:
    module_info.append('SASPP')
if use_ega:
    module_info.append('EGA')
    
print(f"数据集: Stare")
print(f"模型使用模块: {' + '.join(module_info) if module_info else 'UNet(无增强模块)'}")

data_location        = 'Stare/'
training_images_loc  = os.path.join(data_location, 'train/image/')
training_label_loc   = os.path.join(data_location, 'train/label/')
validate_images_loc  = os.path.join(data_location, 'validate/images/')
validate_label_loc   = os.path.join(data_location, 'validate/labels/')

class StareDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

def load_data(files, img_dir, lbl_dir):

    data, labels = [], []
    for fn in files:

        im_path = os.path.join(img_dir, fn)
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
 
        base_name = fn.split('.')[0]  
        lbl_path = os.path.join(lbl_dir, f"{base_name}.png")
        lab = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        
        if im is None or lab is None:
            print(f"警告: 无法读取文件 {fn}")
            print(f"  图像路径: {im_path}")
            print(f"  标签路径: {lbl_path}")
            continue
  
        h, w = im.shape
        th, tw = desired_size, desired_size
   
        top = (th - h)//2
        bottom = th - h - top
        left = (tw - w)//2
        right = tw - w - left
 
        im_pad = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        lbl_pad = cv2.copyMakeBorder(lab, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        im_rz = cv2.resize(im_pad, (tw, th))
        _, lbl_rz = cv2.threshold(cv2.resize(lbl_pad, (tw, th)), 127, 255, cv2.THRESH_BINARY)

        data.append(im_rz.astype('float32')/255.)
        labels.append(lbl_rz.astype('float32')/255.)
    
    if len(data) == 0:
        raise ValueError("没有成功加载任何数据，请检查文件路径和格式")
  
    data = np.array(data).reshape(-1, 1, desired_size, desired_size)
    labels = np.array(labels).reshape(-1, 1, desired_size, desired_size)
    return torch.from_numpy(data), torch.from_numpy(labels)

ckpt_dir = 'Stare/test/checkpoint'
loss_dir = 'Stare/test/loss'
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)

def main():

    print("开始加载Stare数据集...")

    if not os.path.exists(training_images_loc):
        print(f"错误: 训练图像目录不存在: {training_images_loc}")
        return
    if not os.path.exists(validate_images_loc):
        print(f"错误: 验证图像目录不存在: {validate_images_loc}")
        return

    train_files = [f for f in os.listdir(training_images_loc) if f.lower().endswith('.png')]
    val_files = [f for f in os.listdir(validate_images_loc) if f.lower().endswith('.png')]
    
    if len(train_files) == 0:
        print(f"错误: 训练目录中没有找到PNG文件: {training_images_loc}")
        return
    if len(val_files) == 0:
        print(f"错误: 验证目录中没有找到PNG文件: {validate_images_loc}")
        return
    
    print(f"找到训练文件: {len(train_files)} 个")
    print(f"找到验证文件: {len(val_files)} 个")

    try:
        x_train, y_train = load_data(train_files, training_images_loc, training_label_loc)
        x_val, y_val = load_data(val_files, validate_images_loc, validate_label_loc)
        print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    train_loader = DataLoader(StareDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StareDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = UNet(input_channels=1, start_neurons=start_neurons,
                  keep_prob=keep_prob, block_size=block_size,
                  use_saspp=use_saspp, use_ega=use_ega).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    writer = SummaryWriter(log_dir=f'logs/stare_{module_name}')

    weight_path = os.path.join(ckpt_dir, f'UNet_stare_{module_name}.pth')
    if args.restore and os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"恢复权重: {weight_path}")

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print(f"\n开始训练 {epochs} 轮...")
    print("-" * 50)

    for epoch in range(1, epochs+1):

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
  
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}", end='\r')
        
        train_loss = train_loss_sum / train_batches

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                val_loss = criterion(outputs, lbls)
                val_loss_sum += val_loss.item()
                val_batches += 1
        
        val_loss = val_loss_sum / val_batches

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weight_path)
            print(f"  >>> 保存最佳模型: {weight_path} (Val Loss: {val_loss:.4f})")

    writer.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(range(1, epochs+1), val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Stare Dataset - Loss Curve ({module_name.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    loss_curve_path = os.path.join(loss_dir, f'loss_stare_{module_name}.png')
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nLoss 曲线已保存至: {loss_curve_path}")

    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型权重保存在: {weight_path}")
    print("-" * 50)

if __name__ == '__main__':
    main()