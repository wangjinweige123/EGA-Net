import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix
import cv2
from util import crop_to_shape
from model import UNet
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description='评估UNet模型')
parser.add_argument('--use_saspp', action='store_true', help='启用BASCP模块')
parser.add_argument('--use_ega', action='store_true', help='启用EGA模块')
parser.add_argument('--weight_path', type=str, default="Chase/Model/UNet_base.pth", help='模型权重路径')
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
        'data_location': '',
        'result_dir': './Chase/test/result/result-eval',
        'visual_dir': './Chase/test/visualization/visualization-eval',
        'eval_result_file': './Chase/test/evaluation_results_eval.txt'
    }
}

model_suffix = ''
if CONFIG['use_saspp'] and CONFIG['use_ega']:
    model_suffix = '_saspp_ega'
elif CONFIG['use_saspp']:
    model_suffix = '_saspp'
elif CONFIG['use_ega']:
    model_suffix = '_ega'

CONFIG['paths']['result_dir'] = f"./Chase/test/result/result-eval{model_suffix}"
CONFIG['paths']['visual_dir'] = f"./Chase/test/visualization/visualization-eval{model_suffix}"
CONFIG['paths']['eval_result_file'] = f"./Chase/test/evaluation_results_eval{model_suffix}.txt"


CONFIG['paths']['orig_images_loc'] = CONFIG['paths']['data_location'] + 'Chase/test/im/'
CONFIG['paths']['testing_images_loc'] = CONFIG['paths']['data_location'] + 'Chase/test/image/'
CONFIG['paths']['testing_label_loc'] = CONFIG['paths']['data_location'] + 'Chase/test/label/'

modules_used = []
if CONFIG['use_saspp']:
    modules_used.append("BASCP")
if CONFIG['use_ega']:
    modules_used.append("EGA")

print(f"评估模型: UNet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}")
print(f"使用权重文件: {CONFIG['weight_path']}")
print(f"评估结果将保存到: {CONFIG['paths']['eval_result_file']}")
print(f"可视化结果将保存到: {CONFIG['paths']['visual_dir']}")

def setup_directories():

    os.makedirs(CONFIG['paths']['result_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['visual_dir'], exist_ok=True)

def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    return total_params, trainable_params

def calculate_flops(model, input_shape=(1, 1, 1008, 1008)):

    def flop_count_hook(module, input, output):

        if isinstance(module, torch.nn.Conv2d):

            batch_size = input[0].shape[0]
            output_dims = output.shape[2:]
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            filters_per_channel = out_channels // groups
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
            
            active_elements_count = batch_size * int(np.prod(output_dims))
            overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
            
            if module.bias is not None:
                overall_conv_flops += out_channels * active_elements_count
            
            module.__flops__ += overall_conv_flops
            
        elif isinstance(module, torch.nn.BatchNorm2d):

            batch_size = input[0].shape[0]
            output_elements = batch_size * int(np.prod(output.shape[1:]))
            module.__flops__ += 2 * output_elements
            
        elif isinstance(module, torch.nn.ReLU):

            batch_size = input[0].shape[0]
            output_elements = batch_size * int(np.prod(output.shape[1:]))
            module.__flops__ += output_elements
            
        elif isinstance(module, torch.nn.MaxPool2d) or isinstance(module, torch.nn.AvgPool2d):

            batch_size = input[0].shape[0]
            output_elements = batch_size * int(np.prod(output.shape[1:]))
            module.__flops__ += output_elements
            
        elif isinstance(module, torch.nn.ConvTranspose2d):

            batch_size = input[0].shape[0]
            output_dims = output.shape[2:]
            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            
            conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels
            active_elements_count = batch_size * int(np.prod(output_dims))
            overall_conv_flops = conv_per_position_flops * active_elements_count
            
            if module.bias is not None:
                overall_conv_flops += out_channels * active_elements_count
            
            module.__flops__ += overall_conv_flops
    
    def add_flops_counter_variable_or_reset(module):
        if is_supported_instance(module):
            module.__flops__ = 0
    
    def is_supported_instance(module):
        return isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, 
                                  torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.ConvTranspose2d))
    

    model.apply(add_flops_counter_variable_or_reset)
    

    handles = []
    for module in model.modules():
        if is_supported_instance(module):
            handles.append(module.register_forward_hook(flop_count_hook))
    

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
        _ = model(dummy_input)
    

    for handle in handles:
        handle.remove()
    

    total_flops = 0
    for module in model.modules():
        if hasattr(module, '__flops__'):
            total_flops += module.__flops__
    
    return total_flops

def load_test_data():

    test_files = os.listdir(CONFIG['paths']['testing_images_loc'])
    test_data = []
    test_label = []
    file_indices = {}
    
    for i, file_name in enumerate(test_files):
        print(f"处理图像 {i+1}/{len(test_files)}: {file_name}")
        file_indices[i] = file_name
        
        # 读取图像和标签
        im = cv2.imread(CONFIG['paths']['testing_images_loc'] + file_name, cv2.IMREAD_GRAYSCALE)
        label_name = f"Image_{file_name.split('_')[1].split('.')[0]}_1stHO.png"
        label = cv2.imread(CONFIG['paths']['testing_label_loc'] + label_name, cv2.IMREAD_GRAYSCALE)
        
        # 计算需要填充的尺寸
        old_size = im.shape[:2]
        delta_w = CONFIG['desired_size'] - old_size[1]
        delta_h = CONFIG['desired_size'] - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # 填充图像
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # 调整大小
        resized_im = cv2.resize(new_im, (CONFIG['desired_size'], CONFIG['desired_size']))
        temp = cv2.resize(new_label, (CONFIG['desired_size'], CONFIG['desired_size']))
        _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        
        test_data.append(resized_im)
        test_label.append(temp)
    
    # 转换为NumPy数组
    test_data_np = np.array(test_data)
    test_label_np = np.array(test_label)
    
    # 归一化并调整数据维度
    x_test = test_data_np.astype('float32') / 255.
    y_test = test_label_np.astype('float32') / 255.
    x_test = np.reshape(x_test, (len(x_test), 1, CONFIG['desired_size'], CONFIG['desired_size']))
    y_test = np.reshape(y_test, (len(y_test), 1, CONFIG['desired_size'], CONFIG['desired_size']))
    
    return test_data_np, x_test, y_test, file_indices


def load_model():
    """加载模型"""
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
        alt_paths = ["Chase/Model/UNet.pth"]
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
    """执行预测"""
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


def process_results(test_data, y_test, y_pred, file_indices):
    """处理预测结果，创建可视化并计算评估指标"""
    # 获取裁剪形状
    new_shape = (len(y_test), 960, 999, 1)
    
    # 调整数据格式并裁剪
    y_pred_numpy = y_pred.transpose(0, 2, 3, 1)
    y_test_numpy = y_test.transpose(0, 2, 3, 1)
    
    y_pred_cropped = crop_to_shape(y_pred_numpy, new_shape)
    y_test_cropped = crop_to_shape(y_test_numpy, new_shape)
    
    # 裁剪测试数据
    test_data_cropped = []
    for i in range(len(test_data)):
        img_reshaped = np.reshape(test_data[i], (1, test_data[i].shape[0], test_data[i].shape[1], 1))
        img_cropped = crop_to_shape(img_reshaped, (1, new_shape[1], new_shape[2], 1))
        test_data_cropped.append(img_cropped[0, :, :, 0])
    
    print(f"裁剪后预测结果形状: {y_pred_cropped.shape}")
    print(f"裁剪后标签形状: {y_test_cropped.shape}")
    print(f"裁剪后灰度图形状: {test_data_cropped[0].shape}")
    
    # 生成可视化
    y_pred_threshold = np.zeros_like(y_pred_cropped)
    print("\n创建可视化结果...")
    
    create_visualizations(test_data, test_data_cropped, y_pred_cropped, y_test_cropped, 
                          y_pred_threshold, file_indices)
    
    # 评估模型性能
    evaluate_model(y_test_cropped, y_pred_cropped, y_pred_threshold)
    
    return y_test_cropped, y_pred_cropped, y_pred_threshold


def create_visualizations(test_data, test_data_cropped, y_pred_cropped, y_test_cropped, 
                          y_pred_threshold, file_indices):
    """创建和保存可视化结果"""
    for i, y in enumerate(y_pred_cropped):
        current_file = file_indices[i]
        
        # 阈值化预测结果
        y_pred_threshold[i] = (y > 0.5).astype(np.float32)
        
        # 保存预测结果图像
        y_img = (y * 255).astype(np.uint8)
        result_path = f"{CONFIG['paths']['result_dir']}/{i}.png"
        cv2.imwrite(result_path, y_img)
        print(f"已保存预测结果到: {result_path}")
        
        # 获取原始图像
        orig_img = get_original_image(current_file, test_data[i])
        
        # 创建可视化画布
        h, w = y_pred_cropped[i].shape[:2]
        orig_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_AREA)
        
        # 准备各个可视化部分
        gt_img = create_binary_image(y_test_cropped[i, :, :, 0])
        pred_img = create_binary_image(y_pred_threshold[i, :, :, 0])
        overlay_img = create_overlay_image(test_data_cropped[i], 
                                          y_test_cropped[i, :, :, 0], 
                                          y_pred_threshold[i, :, :, 0])
        
        # 组合画布
        canvas = combine_visualizations(orig_img, gt_img, pred_img, overlay_img)
        
        # 保存可视化结果
        vis_output_path = f"{CONFIG['paths']['visual_dir']}/{i}_{current_file.split('.')[0]}_comparison.png"
        cv2.imwrite(vis_output_path, canvas)
        print(f"已保存可视化结果到: {vis_output_path}")


def get_original_image(current_file, default_img):
    """获取原始图像或使用灰度图像代替"""
    base_name = f"Image_{current_file.split('_')[1].split('.')[0]}"
    orig_img_path = os.path.join(CONFIG['paths']['orig_images_loc'], f"{base_name}.png")
    
    if not os.path.exists(orig_img_path):
        for ext in ['.jpg', '.JPG', '.png', '.PNG']:
            alt_path = os.path.join(CONFIG['paths']['orig_images_loc'], f"{base_name}{ext}")
            if os.path.exists(alt_path):
                orig_img_path = alt_path
                break
    
    if os.path.exists(orig_img_path):
        orig_img = cv2.imread(orig_img_path)
        print(f"读取原始图像: {orig_img_path}")
    else:
        print(f"未找到原始图像: {orig_img_path}，使用灰度图像代替")
        orig_img = cv2.cvtColor(default_img, cv2.COLOR_GRAY2BGR)
    
    return orig_img


def create_binary_image(mask):
    """创建二值图像"""
    h, w = mask.shape
    binary_img = np.zeros((h, w, 3), dtype=np.uint8)
    binary_img[mask > 0.5] = [255, 255, 255]
    return binary_img


def create_overlay_image(gray_img, gt_mask, pred_mask):
    """创建叠加图像"""
    # 一定要先将灰度图转为彩色图，保持其背景
    overlay_img = cv2.cvtColor(gray_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # 在血管区域添加颜色
    overlay_img[gt_mask > 0.5, 1] = 255  # 绿色通道表示GT
    overlay_img[pred_mask > 0.5, 2] = 255  # 红色通道表示预测
    
    return overlay_img


def combine_visualizations(orig_img, gt_img, pred_img, overlay_img):
    """组合可视化结果到一个画布"""
    h, w = orig_img.shape[:2]
    title_h = 30
    canvas = np.ones((h + title_h, w * 4, 3), dtype=np.uint8) * 255
    
    # 添加标题
    titles = [
        ('Original Image', w//2 - 80),
        ('Ground Truth', w + w//2 - 80),
        ('Prediction', 2*w + w//2 - 60),
        ('Overlay (GT-green / Pred-red)', 3*w + w//2 - 140)
    ]
    
    for title, pos_x in titles:
        cv2.putText(canvas, title, (pos_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # 放置图像
    canvas[title_h:, 0:w] = orig_img
    canvas[title_h:, w:2*w] = gt_img
    canvas[title_h:, 2*w:3*w] = pred_img
    canvas[title_h:, 3*w:4*w] = overlay_img
    
    return canvas


def evaluate_model(y_test_cropped, y_pred_cropped, y_pred_threshold):
    """计算并记录评估指标"""
    y_test_flat = y_test_cropped.flatten()
    y_pred_threshold_flat = y_pred_threshold.flatten()
    y_pred_flat = y_pred_cropped.flatten()
    
    print(f"扁平化后测试标签长度: {len(y_test_flat)}")
    print(f"扁平化后预测阈值长度: {len(y_pred_threshold_flat)}")
    
    try:
        print("\n计算评估指标...")
        tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_threshold_flat).ravel()
        
        metrics = {
            'sensitivity': recall_score(y_test_flat, y_pred_threshold_flat),
            'specificity': tn / (tn + fp),
            'f1': 2*tp/(2*tp+fn+fp),
            'accuracy': accuracy_score(y_test_flat, y_pred_threshold_flat),
            'auc': roc_auc_score(y_test_flat, y_pred_flat)
        }
        
        print('\n最终评估结果:')
        print('-' * 30)
        print(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}')
        print(f'Specificity (特异度): {metrics["specificity"]:.4f}')
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f'Accuracy (准确率): {metrics["accuracy"]:.4f}')
        print(f'AUC: {metrics["auc"]:.4f}')
        print('-' * 30)
        
        metrics["fps"] = fps
        metrics["time_ms"] = avg_time_ms
        save_evaluation_results(metrics)
      
        
    except Exception as e:
        print("计算评估指标时出错:", e)
        print("测试标签值范围:", np.min(y_test_flat), np.max(y_test_flat))
        print("预测阈值值范围:", np.min(y_pred_threshold_flat), np.max(y_pred_threshold_flat))


def save_evaluation_results(metrics):
    """保存评估结果到文件"""
    # 生成模型名称
    modules_used = []
    if CONFIG['use_saspp']:
        modules_used.append("BASCP")
    if CONFIG['use_ega']:
        modules_used.append("EGA")
    
    model_name = f"UNet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}"
    
    with open(CONFIG['paths']['eval_result_file'], 'w') as f:
        f.write(f'评估模型: {model_name}\n')
        f.write(f'权重文件: {CONFIG["weight_path"]}\n')
        f.write('-' * 50 + '\n')
        f.write(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}\n')
        f.write(f'Specificity (特异度): {metrics["specificity"]:.4f}\n')
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f'Accuracy (准确率): {metrics["accuracy"]:.4f}\n')
        f.write(f'AUC: {metrics["auc"]:.4f}\n')
        f.write('-' * 50 + '\n')
        f.write(f'Params (M): {metrics["params_m"]:.2f}\n')
        f.write(f'FLOPs (G): {metrics["flops_g"]:.2f}\n')
        if "fps" in metrics:
            f.write(f'FPS: {metrics["fps"]:.2f}\n')
        if "time_ms" in metrics:
            f.write(f'Time per image (ms): {metrics["time_ms"]:.2f}\n')
        f.write('-' * 50 + '\n')
    
    print(f"评估结果已保存到: {CONFIG['paths']['eval_result_file']}")


def main():
    """主函数"""
    print(f"使用设备: {CONFIG['device']}")
    
    # 设置目录
    setup_directories()
    
    # 加载数据
    test_data, x_test, y_test, file_indices = load_test_data()
    
    # 加载模型
    model = load_model()
    
    # 计算模型参数量和FLOPs
    print("\n计算模型复杂度...")
    total_params, trainable_params = count_parameters(model)
    params_m = total_params / 1e6  # 转换为百万参数
    
    print("\n计算FLOPs...")
    total_flops = calculate_flops(model, input_shape=(1, 1, CONFIG['desired_size'], CONFIG['desired_size']))
    flops_g = total_flops / 1e9  # 转换为十亿次浮点运算
    
    print(f"模型参数量: {params_m:.2f}M")
    print(f"模型FLOPs: {flops_g:.2f}G")
    
    # 执行预测
    # 执行预测（外层计时，不改变原有推理代码）
    if CONFIG['device'].type == 'cuda':
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    y_pred = predict(model, x_test)
    if CONFIG['device'].type == 'cuda':
        torch.cuda.synchronize()
    avg_time_ms = (time.perf_counter() - _t0) * 1000.0 / len(x_test)
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0


    # 处理结果
    y_test_cropped, y_pred_cropped, y_pred_threshold = process_results(test_data, y_test, y_pred, file_indices)
    
    # 计算评估指标（包含复杂度指标）
    y_test_flat = y_test_cropped.flatten()
    y_pred_threshold_flat = y_pred_threshold.flatten()
    y_pred_flat = y_pred_cropped.flatten()
    
    try:
        print("\n计算最终评估指标...")
        tn, fp, fn, tp = confusion_matrix(y_test_flat, y_pred_threshold_flat).ravel()
        
        metrics = {
            'sensitivity': recall_score(y_test_flat, y_pred_threshold_flat),
            'specificity': tn / (tn + fp),
            'f1': 2*tp/(2*tp+fn+fp),
            'accuracy': accuracy_score(y_test_flat, y_pred_threshold_flat),
            'auc': roc_auc_score(y_test_flat, y_pred_flat),
            'params_m': params_m,
            'flops_g': flops_g
        }
        
        print('\n最终评估结果:')
        print('-' * 50)
        print(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}')
        print(f'Specificity (特异度): {metrics["specificity"]:.4f}')
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f'Accuracy (准确率): {metrics["accuracy"]:.4f}')
        print(f'AUC: {metrics["auc"]:.4f}')
        print(f'Params (M): {metrics["params_m"]:.2f}')
        print(f'FLOPs (G): {metrics["flops_g"]:.2f}')
        print('-' * 50)
        metrics["fps"] = fps
        metrics["time_ms"] = avg_time_ms
        print(f'FPS: {metrics["fps"]:.2f}')
        print(f'Time per image (ms): {metrics["time_ms"]:.2f}')
        save_evaluation_results(metrics)      # <-- 现在写入时能看到 fps/time_ms


        
    except Exception as e:
        print("计算评估指标时出错:", e)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()