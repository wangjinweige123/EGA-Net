import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix
import cv2
from util import crop_to_shape
from model import UNet
import pandas as pd
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description='评估UNet模型 - Stare数据集')
parser.add_argument('--use_saspp', action='store_true', help='启用SASPP模块')
parser.add_argument('--use_ega', action='store_true', help='启用EGA模块')
parser.add_argument('--weight_path', type=str, default="Stare/test/checkpoint/UNet_stare_base.pth", help='模型权重路径')
args = parser.parse_args()

CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'desired_size': 1008,
    'batch_size': 1,
    'weight_path': args.weight_path,
    'use_saspp': args.use_saspp,
    'use_ega': args.use_ega,
    'model_params': {
        'input_channels': 1,
        'start_neurons': 16,
        'keep_prob': 0.87,
        'block_size': 7
    },
    'paths': {
        'data_location': 'Stare/',
        'result_dir': './Stare/test/result/result-eval',
        'visual_dir': './Stare/test/visualization/visualization-eval',
        'eval_result_file': './Stare/test/evaluation_results_eval.txt'
    }
}

model_suffix = ''
if CONFIG['use_saspp'] and CONFIG['use_ega']:
    model_suffix = '_saspp_ega'
elif CONFIG['use_saspp']:
    model_suffix = '_saspp'
elif CONFIG['use_ega']:
    model_suffix = '_ega'

CONFIG['paths']['result_dir'] = f"./Stare/test/result/result-eval{model_suffix}"
CONFIG['paths']['visual_dir'] = f"./Stare/test/visualization/visualization-eval{model_suffix}"
CONFIG['paths']['eval_result_file'] = f"./Stare/test/evaluation_results_eval{model_suffix}.txt"

CONFIG['paths']['orig_images_loc'] = CONFIG['paths']['data_location'] + 'test/im/'
CONFIG['paths']['testing_images_loc'] = CONFIG['paths']['data_location'] + 'test/image/'
CONFIG['paths']['testing_label_loc'] = CONFIG['paths']['data_location'] + 'test/label/'

CONFIG['paths']['threshold_result_file'] = f"./Stare/test/threshold_evaluation_results{model_suffix}.txt"
CONFIG['paths']['threshold_csv_file'] = f"./Stare/test/threshold_detailed_results{model_suffix}.csv"
CONFIG['paths']['threshold_plot_file'] = f"./Stare/test/threshold_curves{model_suffix}.png"

modules_used = []
if CONFIG['use_saspp']:
    modules_used.append("SASPP")
if CONFIG['use_ega']:
    modules_used.append("EGA")

print(f"评估模型: UNet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}")
print(f"数据集: Stare")
print(f"使用权重文件: {CONFIG['weight_path']}")
print(f"评估结果将保存到: {CONFIG['paths']['eval_result_file']}")
print(f"可视化结果将保存到: {CONFIG['paths']['visual_dir']}")


def setup_directories():

    os.makedirs(CONFIG['paths']['result_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['visual_dir'], exist_ok=True)


def load_test_data():

    if not os.path.exists(CONFIG['paths']['testing_images_loc']):
        raise FileNotFoundError(f"测试图像目录不存在: {CONFIG['paths']['testing_images_loc']}")
    
    test_files = [f for f in os.listdir(CONFIG['paths']['testing_images_loc']) 
                  if f.lower().endswith('.png')]
    
    if len(test_files) == 0:
        raise ValueError(f"测试目录中没有找到PNG文件: {CONFIG['paths']['testing_images_loc']}")
    
    test_data = []
    test_label = []
    file_indices = {}
    
    for i, file_name in enumerate(test_files):
        print(f"处理图像 {i+1}/{len(test_files)}: {file_name}")
        file_indices[i] = file_name

        im_path = os.path.join(CONFIG['paths']['testing_images_loc'], file_name)
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

        base_name = file_name.split('.')[0]  # 获取不带扩展名的文件名
        label_name = f"{base_name}.png"  # Stare数据集标签都是png格式
        
        label_path = os.path.join(CONFIG['paths']['testing_label_loc'], label_name)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if im is None or label is None:
            print(f"警告: 无法读取图像或标签文件 {file_name}")
            print(f"  图像路径: {im_path}")
            print(f"  标签路径: {label_path}")
            continue
    
        old_size = im.shape[:2]
        delta_w = CONFIG['desired_size'] - old_size[1]
        delta_h = CONFIG['desired_size'] - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        resized_im = cv2.resize(new_im, (CONFIG['desired_size'], CONFIG['desired_size']))
        temp = cv2.resize(new_label, (CONFIG['desired_size'], CONFIG['desired_size']))
        _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        
        test_data.append(resized_im)
        test_label.append(temp)
    
    if len(test_data) == 0:
        raise ValueError("没有成功加载任何测试数据，请检查文件路径和文件名")

    test_data_np = np.array(test_data)
    test_label_np = np.array(test_label)

    x_test = test_data_np.astype('float32') / 255.
    y_test = test_label_np.astype('float32') / 255.
    x_test = np.reshape(x_test, (len(x_test), 1, CONFIG['desired_size'], CONFIG['desired_size']))
    y_test = np.reshape(y_test, (len(y_test), 1, CONFIG['desired_size'], CONFIG['desired_size']))
    
    return test_data_np, x_test, y_test, file_indices

def load_model():

    model = UNet(input_channels=CONFIG['model_params']['input_channels'],
                   start_neurons=CONFIG['model_params']['start_neurons'],
                   keep_prob=CONFIG['model_params']['keep_prob'],
                   block_size=CONFIG['model_params']['block_size'],
                   use_saspp=CONFIG['use_saspp'],
                   use_ega=CONFIG['use_ega'])
    
    if os.path.isfile(CONFIG['weight_path']):
        model.load_state_dict(torch.load(CONFIG['weight_path'], map_location=CONFIG['device']))
        print(f"加载模型权重: {CONFIG['weight_path']}")
    else:
        print(f"警告: 找不到权重文件 {CONFIG['weight_path']}")
        # 尝试不同的权重文件路径
        alt_paths = [
            "Stare/test/checkpoint/UNet_stare_base.pth",
            "Stare/test/checkpoint/UNet_stare_saspp.pth",
            "Stare/test/checkpoint/UNet_stare_ega.pth",
            "Stare/test/checkpoint/UNet_stare_saspp_ega.pth"
        ]
        for alt_path in alt_paths:
            if os.path.isfile(alt_path):
                model.load_state_dict(torch.load(alt_path, map_location=CONFIG['device']))
                print(f"加载替代模型权重: {alt_path}")
                break
        else:
            print("无法找到任何模型权重文件，使用随机初始化的模型")
    
    model.to(CONFIG['device'])
    model.eval()
    return model


def predict(model, x_test):
    x_test_tensor = torch.from_numpy(x_test).to(CONFIG['device'])
    y_pred_list = []
    
    print("\n开始模型预测...")
    with torch.no_grad():
        for i in range(0, len(x_test_tensor), CONFIG['batch_size']):
            batch_end = min(i + CONFIG['batch_size'], len(x_test_tensor))
            input_batch = x_test_tensor[i:batch_end]
            output = model(input_batch)
            y_pred_list.append(output.cpu().numpy())
            print(f"已处理 {batch_end}/{len(x_test_tensor)} 个样本", end="\r")
    
    print("\n预测完成！")
    return np.concatenate(y_pred_list, axis=0)


def evaluate_at_threshold(y_true_flat, y_pred_flat, threshold):
    y_pred_threshold_flat = (y_pred_flat > threshold).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_threshold_flat).ravel()
        
        sensitivity = recall_score(y_true_flat, y_pred_threshold_flat)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2*tp/(2*tp+fn+fp) if (2*tp+fn+fp) > 0 else 0
        accuracy = accuracy_score(y_true_flat, y_pred_threshold_flat)
        
        return {
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    except Exception as e:
        print(f"阈值 {threshold} 计算指标时出错: {e}")
        return {
            'threshold': threshold,
            'sensitivity': 0,
            'specificity': 0,
            'f1': 0,
            'accuracy': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }


def multi_threshold_evaluation(y_test_cropped, y_pred_cropped):

    print("\n=== 开始多阈值评估 ===")

    y_test_flat = y_test_cropped.flatten()
    y_pred_flat = y_pred_cropped.flatten()
    
    try:
        auc = roc_auc_score(y_test_flat, y_pred_flat)
    except Exception as e:
        print(f"计算AUC时出错: {e}")
        auc = 0

    thresholds = np.arange(0.1, 1.1, 0.1)

    results = []
    
    print("评估不同阈值下的性能...")
    for threshold in thresholds:
        print(f"  阈值: {threshold:.1f}")
        metrics = evaluate_at_threshold(y_test_flat, y_pred_flat, threshold)
        metrics['auc'] = auc  # 添加AUC值
        results.append(metrics)

    df_results = pd.DataFrame(results)

    best_f1_idx = df_results['f1'].idxmax()
    best_threshold = df_results.loc[best_f1_idx, 'threshold']
    best_metrics = df_results.loc[best_f1_idx]
    
    print(f"\n=== 最优阈值结果 (F1最高) ===")
    print(f"最优阈值: {best_threshold:.1f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"Specificity: {best_metrics['specificity']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"AUC: {best_metrics['auc']:.4f}")
    print("=" * 35)

    save_threshold_results(df_results, best_metrics, best_threshold)
    
    return df_results, best_metrics, best_threshold


def save_threshold_results(df_results, best_metrics, best_threshold):

    modules_used = []
    if CONFIG['use_saspp']:
        modules_used.append("SASPP")
    if CONFIG['use_ega']:
        modules_used.append("EGA")
    
    model_name = f"UNet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}"

    df_results.to_csv(CONFIG['paths']['threshold_csv_file'], index=False, float_format='%.4f')
    print(f"详细阈值结果已保存到: {CONFIG['paths']['threshold_csv_file']}")

    with open(CONFIG['paths']['threshold_result_file'], 'w', encoding='utf-8') as f:
        f.write(f'阈值评估模型: {model_name}\n')
        f.write(f'数据集: Stare\n')
        f.write(f'权重文件: {CONFIG["weight_path"]}\n')
        f.write('=' * 50 + '\n')
        f.write(f'最优阈值 (F1最高): {best_threshold:.1f}\n')
        f.write('-' * 30 + '\n')
        f.write(f'F1 Score: {best_metrics["f1"]:.4f}\n')
        f.write(f'Sensitivity (敏感度/召回率): {best_metrics["sensitivity"]:.4f}\n')
        f.write(f'Specificity (特异度): {best_metrics["specificity"]:.4f}\n')
        f.write(f'Accuracy (准确率): {best_metrics["accuracy"]:.4f}\n')
        f.write(f'AUC: {best_metrics["auc"]:.4f}\n')
        f.write('=' * 50 + '\n')
        f.write('\n详细阈值结果:\n')
        f.write('-' * 80 + '\n')
        f.write(f'{"阈值":<8} {"F1":<8} {"Sensitivity":<12} {"Specificity":<12} {"Accuracy":<10} {"AUC":<8}\n')
        f.write('-' * 80 + '\n')
        for _, row in df_results.iterrows():
            f.write(f'{row["threshold"]:<8.1f} {row["f1"]:<8.4f} {row["sensitivity"]:<12.4f} '
                   f'{row["specificity"]:<12.4f} {row["accuracy"]:<10.4f} {row["auc"]:<8.4f}\n')
    
    print(f"阈值评估结果已保存到: {CONFIG['paths']['threshold_result_file']}")

    plot_threshold_curves(df_results)

def plot_threshold_curves(df_results):

    plt.figure(figsize=(12, 8))

    plt.plot(df_results['threshold'], df_results['f1'], 'o-', label='F1 Score', linewidth=2, markersize=6)
    plt.plot(df_results['threshold'], df_results['sensitivity'], 's-', label='Sensitivity', linewidth=2, markersize=6)
    plt.plot(df_results['threshold'], df_results['specificity'], '^-', label='Specificity', linewidth=2, markersize=6)
    plt.plot(df_results['threshold'], df_results['accuracy'], 'd-', label='Accuracy', linewidth=2, markersize=6)

    best_f1_idx = df_results['f1'].idxmax()
    best_threshold = df_results.loc[best_f1_idx, 'threshold']
    best_f1 = df_results.loc[best_f1_idx, 'f1']
    plt.plot(best_threshold, best_f1, 'ro', markersize=10, label=f'Best F1 (threshold={best_threshold:.1f})')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Stare Dataset: Performance Metrics vs Threshold', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.05, 1.05)
    plt.ylim(0, 1)

    plt.xticks(np.arange(0.1, 1.1, 0.1))

    plt.savefig(CONFIG['paths']['threshold_plot_file'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"阈值曲线图已保存到: {CONFIG['paths']['threshold_plot_file']}")


def process_results(test_data, y_test, y_pred, file_indices):

    new_shape = (len(y_test), 605, 700, 1)  # 注意：Stare是700×605，所以height=605, width=700

    y_pred_numpy = y_pred.transpose(0, 2, 3, 1)
    y_test_numpy = y_test.transpose(0, 2, 3, 1)
    
    y_pred_cropped = crop_to_shape(y_pred_numpy, new_shape)
    y_test_cropped = crop_to_shape(y_test_numpy, new_shape)

    test_data_cropped = []
    for i in range(len(test_data)):
        img_reshaped = np.reshape(test_data[i], (1, test_data[i].shape[0], test_data[i].shape[1], 1))
        img_cropped = crop_to_shape(img_reshaped, (1, new_shape[1], new_shape[2], 1))
        test_data_cropped.append(img_cropped[0, :, :, 0])
    
    print(f"裁剪后预测结果形状: {y_pred_cropped.shape}")
    print(f"裁剪后标签形状: {y_test_cropped.shape}")
    print(f"裁剪后灰度图形状: {test_data_cropped[0].shape}")

    df_results, best_metrics, best_threshold = multi_threshold_evaluation(y_test_cropped, y_pred_cropped)

    y_pred_threshold = np.zeros_like(y_pred_cropped)
    print(f"\n使用最优阈值 {best_threshold:.1f} 创建可视化结果...")
    
    create_visualizations(test_data, test_data_cropped, y_pred_cropped, y_test_cropped, 
                          y_pred_threshold, file_indices, best_threshold)

    evaluate_model(y_test_cropped, y_pred_cropped, y_pred_threshold, best_threshold, best_metrics)
    
    return y_test_cropped, y_pred_cropped, y_pred_threshold

def create_visualizations(test_data, test_data_cropped, y_pred_cropped, y_test_cropped, 
                          y_pred_threshold, file_indices, best_threshold):

    for i, y in enumerate(y_pred_cropped):
        current_file = file_indices[i]

        y_pred_threshold[i] = (y > best_threshold).astype(np.float32)

        y_img = (y * 255).astype(np.uint8)
        result_path = f"{CONFIG['paths']['result_dir']}/{i}.png"
        cv2.imwrite(result_path, y_img)
        print(f"已保存预测结果到: {result_path}")

        orig_img = get_original_image(current_file, test_data[i])

        h, w = y_pred_cropped[i].shape[:2]
        orig_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_AREA)

        gt_img = create_binary_image(y_test_cropped[i, :, :, 0])
        pred_img = create_binary_image(y_pred_threshold[i, :, :, 0])
        overlay_img = create_overlay_image(test_data_cropped[i], 
                                          y_test_cropped[i, :, :, 0], 
                                          y_pred_threshold[i, :, :, 0])

        canvas = combine_visualizations(orig_img, gt_img, pred_img, overlay_img)

        vis_output_path = f"{CONFIG['paths']['visual_dir']}/{i}_{current_file.split('.')[0]}_comparison.png"
        cv2.imwrite(vis_output_path, canvas)
        print(f"已保存可视化结果到: {vis_output_path}")

def get_original_image(current_file, default_img):

    base_name = current_file.split('.')[0]  # 获取不带扩展名的文件名

    orig_img_path = os.path.join(CONFIG['paths']['orig_images_loc'], f"{base_name}.PNG")

    if not os.path.exists(orig_img_path):
        for ext in ['.png', '.PNG', '.jpg', '.JPG']:
            alt_path = os.path.join(CONFIG['paths']['orig_images_loc'], f"{base_name}{ext}")
            if os.path.exists(alt_path):
                orig_img_path = alt_path
                break
    
    if os.path.exists(orig_img_path):
        orig_img = cv2.imread(orig_img_path)
        print(f"读取原始图像: {orig_img_path}")
        return orig_img
    else:
        print(f"未找到原始图像，使用灰度图像代替")
        orig_img = cv2.cvtColor(default_img, cv2.COLOR_GRAY2BGR)
        return orig_img


def create_binary_image(mask):

    h, w = mask.shape
    binary_img = np.zeros((h, w, 3), dtype=np.uint8)
    binary_img[mask > 0.5] = [255, 255, 255]
    return binary_img


def create_overlay_image(gray_img, gt_mask, pred_mask):

    overlay_img = cv2.cvtColor(gray_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

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

def evaluate_model(y_test_cropped, y_pred_cropped, y_pred_threshold, best_threshold, best_metrics):

    y_test_flat = y_test_cropped.flatten()
    y_pred_threshold_flat = y_pred_threshold.flatten()
    y_pred_flat = y_pred_cropped.flatten()
    
    print(f"扁平化后测试标签长度: {len(y_test_flat)}")
    print(f"扁平化后预测阈值长度: {len(y_pred_threshold_flat)}")
    
    try:
        print(f"\n计算评估指标 (使用最优阈值 {best_threshold:.1f})...")
        
        metrics = {
            'threshold': best_threshold,
            'sensitivity': best_metrics['sensitivity'],
            'specificity': best_metrics['specificity'],
            'f1': best_metrics['f1'],
            'accuracy': best_metrics['accuracy'],
            'auc': best_metrics['auc']
        }
        
        print('\n最终评估结果 (Stare数据集):')
        print('-' * 40)
        print(f'最优阈值: {metrics["threshold"]:.1f}')
        print(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}')
        print(f'Specificity (特异度): {metrics["specificity"]:.4f}')
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f'Accuracy (准确率): {metrics["accuracy"]:.4f}')
        print(f'AUC: {metrics["auc"]:.4f}')
        print('-' * 40)
        
        save_evaluation_results(metrics)
        
    except Exception as e:
        print("计算评估指标时出错:", e)
        print("测试标签值范围:", np.min(y_test_flat), np.max(y_test_flat))
        print("预测阈值值范围:", np.min(y_pred_threshold_flat), np.max(y_pred_threshold_flat))


def save_evaluation_results(metrics):

    modules_used = []
    if CONFIG['use_saspp']:
        modules_used.append("SASPP")
    if CONFIG['use_ega']:
        modules_used.append("EGA")
    model_name = f"UNet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}"

    with open(CONFIG['paths']['eval_result_file'], 'w', encoding='utf-8') as f:
        f.write(f'评估模型: {model_name}\n')
        f.write(f'数据集: Stare\n')
        f.write(f'权重文件: {CONFIG["weight_path"]}\n')
        f.write('-' * 30 + '\n')
        f.write(f'最优阈值: {metrics["threshold"]:.1f}\n')
        f.write(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}\n')
        f.write(f'Specificity (特异度): {metrics["specificity"]:.4f}\n')
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f'Accuracy (准确率): {metrics["accuracy"]:.4f}\n')
        f.write(f'AUC: {metrics["auc"]:.4f}\n')
        f.write('-' * 30 + '\n')

        if 'params_m' in CONFIG:
            f.write(f'Params (M): {CONFIG["params_m"]:.2f}\n')
        if 'flops_g' in CONFIG:
            f.write(f'FLOPs (G): {CONFIG["flops_g"]:.2f}\n')
        if 'fps' in CONFIG:
            f.write(f'FPS: {CONFIG["fps"]:.2f}\n')
        if 'time_ms' in CONFIG:
            f.write(f'Time per image (ms): {CONFIG["time_ms"]:.2f}\n')

    print(f"评估结果已保存到: {CONFIG['paths']['eval_result_file']}")


def count_parameters(model):
    """计算模型参数量（总量与可训练量）"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def calculate_flops(model, input_shape):

    def is_supported_instance(m):
        return isinstance(m, (
            torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU,
            torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.ConvTranspose2d
        ))

    def add_flops_var_or_reset(m):
        if is_supported_instance(m):
            m.__flops__ = 0

    def flop_count_hook(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            b = inputs[0].shape[0]
            k_h, k_w = module.kernel_size
            in_ch = module.in_channels
            out_ch = module.out_channels
            groups = module.groups
            out_h, out_w = output.shape[2], output.shape[3]
            conv_per_pos = k_h * k_w * (in_ch // groups)
            flops = b * out_h * out_w * conv_per_pos * out_ch
            if module.bias is not None:
                flops += b * out_h * out_w * out_ch
            module.__flops__ += flops

        elif isinstance(module, torch.nn.BatchNorm2d):
            b = inputs[0].shape[0]
            out_elems = b * int(np.prod(output.shape[1:]))
            module.__flops__ += 2 * out_elems

        elif isinstance(module, torch.nn.ReLU):
            b = inputs[0].shape[0]
            out_elems = b * int(np.prod(output.shape[1:]))
            module.__flops__ += out_elems

        elif isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
            b = inputs[0].shape[0]
            out_elems = b * int(np.prod(output.shape[1:]))
            module.__flops__ += out_elems

        elif isinstance(module, torch.nn.ConvTranspose2d):
            b = inputs[0].shape[0]
            k_h, k_w = module.kernel_size
            in_ch = module.in_channels
            out_ch = module.out_channels
            out_h, out_w = output.shape[2], output.shape[3]
            conv_per_pos = k_h * k_w * in_ch
            flops = b * out_h * out_w * conv_per_pos
            if module.bias is not None:
                flops += b * out_h * out_w * out_ch
            module.__flops__ += flops

    model.apply(add_flops_var_or_reset)
    handles = []
    for m in model.modules():
        if is_supported_instance(m):
            handles.append(m.register_forward_hook(flop_count_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=next(model.parameters()).device)
        _ = model(dummy)

    total_flops = 0
    for m in model.modules():
        if hasattr(m, "__flops__"):
            total_flops += m.__flops__
    for h in handles:
        h.remove()

    return total_flops


@torch.no_grad()
def measure_fps(model, in_ch: int, size: int, device, warmup=20, iters=100):
    model.eval()
    # 统一成 torch.device
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    x = torch.randn(1, in_ch, size, size, device=dev)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
        for _ in range(warmup):
            _ = model(x)

        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # 毫秒

        times.sort()
        l = int(len(times) * 0.1)
        r = int(len(times) * 0.9)
        trimmed = times[l:r] if r > l else times
        avg_ms = float(np.mean(trimmed))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        return fps, avg_ms
    else:

        for _ in range(5):
            _ = model(x)
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            _  = model(x)
            t1 = time.perf_counter()
            ts.append((t1 - t0) * 1000.0)
        ts.sort()
        avg_ms = float(np.mean(ts[2:-2])) if len(ts) > 4 else float(np.mean(ts))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        return fps, avg_ms


def main():

    print(f"使用设备: {CONFIG['device']}")

    setup_directories()

    test_data, x_test, y_test, file_indices = load_test_data()

    model = load_model()

    print("\n计算模型复杂度指标（使用672×672，接近Stare的700×605）...")
    total_params, _ = count_parameters(model)
    in_ch = CONFIG['model_params']['input_channels']

    flops_side = 672
    total_flops = calculate_flops(model, input_shape=(1, in_ch, flops_side, flops_side))
    params_m = total_params / 1e6
    flops_g = total_flops / 1e9
    
    print(f"模型参数量: {params_m:.2f}M")
    print(f"模型FLOPs (672×672): {flops_g:.2f}G")

    print("\n测量推理速度（使用672×672）...")
    fps_mbf, avg_ms_mbf = measure_fps(
        model,
        in_ch=in_ch,
        size=672, 
        device=CONFIG['device'],
        warmup=20,
        iters=100
    )
    fps = fps_mbf
    avg_time_ms = avg_ms_mbf
    
    print(f"FPS: {fps:.2f}")
    print(f"Time per image (ms): {avg_time_ms:.2f}")

    CONFIG['params_m'] = params_m
    CONFIG['flops_g'] = flops_g
    CONFIG['fps'] = fps
    CONFIG['time_ms'] = avg_time_ms

    y_pred = predict(model, x_test)

    process_results(test_data, y_test, y_pred, file_indices)
    
    print("\n评估完成！")
    print(f"阈值评估结果文件: {CONFIG['paths']['threshold_result_file']}")
    print(f"详细阈值CSV文件: {CONFIG['paths']['threshold_csv_file']}")
    print(f"阈值曲线图文件: {CONFIG['paths']['threshold_plot_file']}")


if __name__ == "__main__":
    main()