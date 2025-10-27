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

# 命令行参数解析
parser = argparse.ArgumentParser(description='UNet Training with/without SASPP and EGA')
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
    
print(f"模型使用模块: {' + '.join(module_info) if module_info else 'UNet(无增强模块)'}")


data_location        = ''
training_images_loc  = os.path.join(data_location, 'Chase/train/image/')
training_label_loc   = os.path.join(data_location, 'Chase/train/label/')
validate_images_loc  = os.path.join(data_location, 'Chase/validate/images/')
validate_label_loc   = os.path.join(data_location, 'Chase/validate/labels/')


class ChaseDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

def load_data(files, img_dir, lbl_dir):
    data, labels = [], []
    for fn in files:
        im = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(os.path.join(lbl_dir, f"Image_{fn.split('_')[1].split('.')[0]}_1stHO.png"), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        th, tw = desired_size, desired_size
        top = (th - h)//2; bottom = th - h - top
        left = (tw - w)//2; right = tw - w - left
        im_pad = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        lbl_pad = cv2.copyMakeBorder(lab, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        im_rz = cv2.resize(im_pad, (tw, th))
        _, lbl_rz = cv2.threshold(cv2.resize(lbl_pad, (tw, th)), 127, 255, cv2.THRESH_BINARY)
        data.append(im_rz.astype('float32')/255.)
        labels.append(lbl_rz.astype('float32')/255.)
    data = np.array(data).reshape(-1,1,desired_size,desired_size)
    labels = np.array(labels).reshape(-1,1,desired_size,desired_size)
    return torch.from_numpy(data), torch.from_numpy(labels)

ckpt_dir = 'Chase/test/checkpoint'
loss_dir = 'Chase/test/loss'
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)

def main():
    train_files = os.listdir(training_images_loc)
    val_files   = os.listdir(validate_images_loc)
    x_train, y_train = load_data(train_files, training_images_loc, training_label_loc)
    x_val,   y_val   = load_data(val_files,   validate_images_loc, validate_label_loc)
    print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}")

    train_loader = DataLoader(ChaseDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ChaseDataset(x_val,   y_val),   batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(input_channels=1, start_neurons=start_neurons,
                  keep_prob=keep_prob, block_size=block_size,
                  use_saspp=use_saspp, use_ega=use_ega).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=f'logs/{module_name}')

    weight_path = os.path.join(ckpt_dir, f'UNet_{module_name}.pth')
    if args.restore and os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print(f"恢复权重: {weight_path}")

    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        tloss = tacc = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        train_loss = tloss / len(train_loader)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                outv = model(imgs.to(device))
                vloss += criterion(outv, lbls.to(device)).item()
        val_loss = vloss / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss',   val_loss,   epoch)

        if val_loss < best_val_acc or epoch == 1:
            best_val_acc = val_loss
            torch.save(model.state_dict(), weight_path)
            print(f"保存模型: {weight_path}")

    writer.close()

    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve ({module_name.upper()})')
    plt.legend()
    plt.savefig(os.path.join(loss_dir, f'loss_{module_name}.png'))
    plt.close()
    print(f"Loss 曲线已保存至 {os.path.join(loss_dir, f'loss_{module_name}.png')}")

    print("训练完成！")

if __name__ == '__main__':
    main()