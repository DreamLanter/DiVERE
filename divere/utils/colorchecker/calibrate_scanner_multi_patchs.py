#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多图像联合扫描仪校准脚本

功能:
- 输入一个或多个经过裁切的ColorChecker 24色卡照片。
- 自动提取所有照片中24个色块的颜色值。
- 与内置的标准ColorChecker Lab颜色值进行比较。
- 计算并拟合出一个共享的校正矩阵M，以及每张照片独立的偏移向量g_i。
- 将结果保存到一个JSON文件中。

关系式: D_ref = D_img_i @ M + g_i
  - D_ref: 来自标准参考数据的密度值 (负值)。
  - D_img_i: 来自第i张扫描照片的密度值 (正值)。
  - M: 所有照片共享的颜色转换矩阵。
  - g_i: 第i张照片独立的偏移向量。

用法:
  python calibrate_scanner_multi_patchs.py [输出JSON路径] [输入照片1] [输入照片2] ...
  
示例:
  python calibrate_scanner_multi_patchs.py ./config/matrices/my_scanner_matrix.json ./scans/scan1.tif ./scans/scan2.tif
"""

import numpy as np
import cv2
import json
import argparse
from scipy.optimize import minimize
from pathlib import Path
import sys

# --- 部分 1: 内置数据和参考颜色处理函数 (与单图像脚本相同) ---

def get_reference_lab_data():
    """返回内置的标准ColorChecker 24色块Lab数据 (D50, 2度观察者)。"""
    return {
        'A1': [37.54, 14.37, 14.92], 'A2': [65.06, 19.46, 17.26], 'A3': [49.99, -4.73, -22.37],
        'A4': [43.37, -12.42, 22.28], 'A5': [54.81, 8.57, -25.50], 'A6': [70.82, -32.28, -0.12],
        'B1': [61.51, 36.06, 57.05], 'B2': [40.01, 10.35, -45.02], 'B3': [50.97, 48.49, 16.54],
        'B4': [30.10, 22.54, -20.87], 'B5': [72.02, -23.70, 58.19], 'B6': [71.35, 18.43, 67.82],
        'C1': [29.26, 13.08, -49.91], 'C2': [54.23, -38.34, 31.33], 'C3': [42.06, 55.22, 28.11],
        'C4': [81.94, 4.13, 80.72], 'C5': [51.13, 50.41, -14.13], 'C6': [49.57, -29.71, -28.32],
        'D1': [95.19, -1.03, 2.93], 'D2': [81.29, -0.57, 0.44], 'D3': [66.89, -0.75, -0.06],
        'D4': [50.76, -0.13, 0.14], 'D5': [35.63, -0.46, -0.48], 'D6': [20.64, 0.07, -0.46]
    }

def lab_to_xyz(lab):
    """将Lab值(D50)转换为XYZ值。"""
    L, a, b = lab
    Xn, Yn, Zn = 0.96422, 1.00000, 0.82521
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    def f_inv(f):
        f_cubed = f ** 3
        return f_cubed if f_cubed > 0.008856 else (f - 16/116) / 7.787
    X, Y, Z = Xn * f_inv(fx), Yn * f_inv(fy), Zn * f_inv(fz)
    return [X, Y, Z]

def chromatic_adaptation(xyz_d50):
    """使用Bradford变换将XYZ从D50调整到D60 (ACEScg的白点)。"""
    M_A = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
    M_A_inv = np.linalg.inv(M_A)
    XYZ_D50, XYZ_D60 = np.array([0.96422, 1.0, 0.82521]), np.array([0.95255, 1.0, 1.00882])
    rgb_cone_d50, rgb_cone_d60 = M_A @ XYZ_D50, M_A @ XYZ_D60
    adaptation_matrix = M_A_inv @ np.diag(rgb_cone_d60 / rgb_cone_d50) @ M_A
    return (adaptation_matrix @ np.array(xyz_d50)).tolist()

def xyz_to_acescg_rgb(xyz):
    """将XYZ值(D60)转换为ACEScg RGB值。"""
    xyz_to_ap1_matrix = np.array([[1.64102338,-0.32480335,-0.23642469],[-0.66366286,1.6153316,0.01675635],[0.0117219,-0.00828444,0.98839486]])
    return (xyz_to_ap1_matrix @ np.array(xyz)).tolist()

def process_reference_data():
    """处理内置的Lab数据，计算并返回每个色块的密度值。"""
    print("Step 1: Processing internal reference ColorChecker data...")
    lab_data = get_reference_lab_data()
    reference_densities = {}
    for patch_id, lab in lab_data.items():
        xyz_d50 = lab_to_xyz(lab)
        xyz_d60 = chromatic_adaptation(xyz_d50)
        rgb = xyz_to_acescg_rgb(xyz_d60)
        density = np.log10(np.clip(rgb, 1e-10, 1.0)).tolist()
        reference_densities[patch_id] = density
    print("✓ Reference densities calculated for 24 patches.")
    return reference_densities

# --- 部分 2: 扫描图像处理函数 (与单图像脚本相同) ---

COLORCHECKER_LAYOUT = {
    'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3), 'A5': (0, 4), 'A6': (0, 5),
    'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3), 'B5': (1, 4), 'B6': (1, 5),
    'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3), 'C5': (2, 4), 'C6': (2, 5),
    'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3), 'D5': (3, 4), 'D6': (3, 5)
}

def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError(f"无法加载图像: {image_path}")
    if len(image.shape) == 3 and image.shape[2] == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / (65535.0 if np.max(image) > 255 else 255.0)
    return np.clip(image, 0.0, 1.0)

def extract_patches_from_image(image, patch_size_ratio=0.3):
    height, width, _ = image.shape
    patch_height, patch_width = height // 4, width // 6
    center_height, center_width = int(patch_height * patch_size_ratio), int(patch_width * patch_size_ratio)
    patches = {}
    for patch_id, (row, col) in COLORCHECKER_LAYOUT.items():
        y_start, x_start = row * patch_height, col * patch_width
        center_y_start, center_x_start = y_start + (patch_height-center_height)//2, x_start + (patch_width-center_width)//2
        center_patch = image[center_y_start:center_y_start+center_height, center_x_start:center_x_start+center_width]
        patches[patch_id] = np.mean(center_patch, axis=(0, 1)).tolist()
    return patches

def process_image_data(image_path):
    print(f"\nProcessing image '{Path(image_path).name}'...")
    image = load_image(image_path)
    patch_rgbs = extract_patches_from_image(image)
    image_densities = {pid: (-np.log10(np.clip(rgb, 1e-10, 1.0))).tolist() for pid, rgb in patch_rgbs.items()}
    print("✓ Image densities calculated.")
    return image_densities

# --- 部分 3: 矩阵拟合函数 (新逻辑) ---

def prepare_data_for_fitting(ref_densities, all_img_densities, gray_patch_weight, skin_tone_weight):
    """准备所有图像的数据用于联合优化。"""
    print("\nStep 2: Preparing combined data for joint optimization...")
    D_ref_list, D_img_list, weights_list = [], [], []
    image_indices = []
    
    start_index = 0
    for img_path, img_densities in all_img_densities.items():
        num_patches_in_image = len(img_densities)
        
        for patch_id in sorted(img_densities.keys()):
            if patch_id in ref_densities:
                D_ref_list.append(ref_densities[patch_id])
                D_img_list.append(img_densities[patch_id])
                if patch_id in ['D2', 'D3', 'D4', 'D5', 'D6']:
                    weights_list.append(gray_patch_weight)
                elif patch_id in ['A1', 'A2']:
                    weights_list.append(skin_tone_weight)
                else:
                    weights_list.append(1.0)
        
        image_indices.append({'path': img_path, 'start': start_index, 'end': start_index + num_patches_in_image})
        start_index += num_patches_in_image

    D_ref, D_img = np.array(D_ref_list), np.array(D_img_list)
    weights = np.array(weights_list).reshape(-1, 1)
    
    print(f"✓ Combined data prepared for {len(all_img_densities)} images.")
    return D_ref, D_img, weights, image_indices

def objective_function(params, D_ref, D_img, weights, image_indices):
    """联合优化的目标函数。模型: D_ref^T = M @ D_img^T + b_i * g^T"""
    num_images = len(image_indices)
    M = params[:9].reshape(3, 3)
    g = params[9:12]
    b_scalars = params[12:]
    
    total_error = 0
    g_col = g.reshape(-1, 1) # Shape (3, 1) for broadcasting

    for i, img_info in enumerate(image_indices):
        start, end = img_info['start'], img_info['end']
        D_img_i, D_ref_i, weights_i = D_img[start:end], D_ref[start:end], weights[start:end]
        b_i = b_scalars[i]
        
        # D_img_i shape: (24, 3) -> D_img_i.T shape: (3, 24)
        # D_pred_i.T shape: (3, 24)
        D_pred_i_T = M @ D_img_i.T + b_i * g_col
        
        # D_pred_i shape: (24, 3)
        D_pred_i = D_pred_i_T.T
        
        error_i = np.sum(((D_ref_i - D_pred_i) ** 2) * weights_i)
        total_error += error_i
        
    return total_error / len(D_ref)

def fit_joint_matrix(D_ref, D_img, weights, image_indices):
    """执行联合拟合操作。"""
    print("\nStep 3: Fitting joint density relationship matrix...")
    num_images = len(image_indices)
    
    # 初始猜测: M为-1单位矩阵, g为零向量, 所有b_i标量为1.0
    M_initial = np.array([-1., 0., 0., 0., -1., 0., 0., 0., -1.])
    g_initial = np.zeros(3)
    b_initial = np.ones(num_images)
    initial_params = np.concatenate([M_initial, g_initial, b_initial])
    
    result = minimize(
        objective_function, initial_params,
        args=(D_ref, D_img, weights, image_indices),
        method='L-BFGS-B', options={'maxiter': 20000}
    )
    
    if not result.success:
        print(f"Warning: Optimization may not have converged. Message: {result.message}")
        
    M = result.x[:9].reshape(3, 3)
    g = result.x[9:12]
    b_scalars = result.x[12:]
    mse = result.fun
    
    print(f"✓ Joint fitting complete. Final Mean Squared Error: {mse:.6f}")
    return M, g, b_scalars, mse

def normalize_matrix(M):
    M11 = M[0, 0]
    if abs(M11) < 1e-10: return M, M11
    return M / M11, M11

# --- 部分 4: 主函数 ---

def main():
    """主执行函数。"""
    parser = argparse.ArgumentParser(
        description='从一个或多个ColorChecker照片中计算联合校准矩阵。',
        epilog="示例:\n  python calibrate_scanner_multi_patchs.py ./out.json ./s1.tif ./s2.tif"
    )
    parser.add_argument('output_json', help='输出的校准矩阵JSON文件路径。')
    parser.add_argument('input_images', nargs='+', help='一个或多个输入的ColorChecker图像文件路径。')
    parser.add_argument('--gray-weight', type=float, default=20.0, help='灰阶色块(D2-D6)的优化权重。')
    parser.add_argument('--skin-weight', type=float, default=20.0, help='肤色色块(A1, A2)的优化权重。')
    args = parser.parse_args()
    
    try:
        ref_densities = process_reference_data()
        
        all_img_densities = {path: process_image_data(path) for path in args.input_images}
        
        D_ref, D_img, weights, image_indices = prepare_data_for_fitting(
            ref_densities, all_img_densities, args.gray_weight, args.skin_weight
        )
        
        M, g, b_scalars, mse = fit_joint_matrix(D_ref, D_img, weights, image_indices)
        
        M_normalized, M11 = normalize_matrix(M)
        
        b_scalars_by_image = {Path(info['path']).name: b_scalars[i] for i, info in enumerate(image_indices)}
        
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'description': 'Joint scanner calibration with shared M, shared g, and per-image scalar b_i.',
            'relationship': 'D_reference^T = M @ D_image_i^T + b_i * g^T',
            'M_matrix': M.tolist(),
            'M_normalized': M_normalized.tolist(),
            'M11_value': M11,
            'g_vector': g.tolist(),
            'b_scalars_by_image': b_scalars_by_image,
            'mean_squared_error': mse
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Joint calibration successful! Results saved to:\n{output_path.resolve()}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

