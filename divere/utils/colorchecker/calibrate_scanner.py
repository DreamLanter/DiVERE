#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一扫描仪校准脚本

功能:
- 输入一张经过裁切的ColorChecker 24色卡照片。
- 自动提取24个色块的颜色值。
- 与内置的标准ColorChecker Lab颜色值进行比较。
- 计算并拟合出校正矩阵M和偏移向量g。
- 将结果保存到一个JSON文件中。

关系式: D_ref = D_img @ M + g
  - D_ref: 来自标准参考数据的密度值 (负值)。
  - D_img: 来自扫描照片的密度值 (正值)。

用法:
  python calibrate_scanner.py [输入照片路径] [输出JSON路径]
  
示例:
  python calibrate_scanner.py ./test_scans/CC0-Croped.tif ./config/matrices/my_scanner_matrix.json
"""

import numpy as np
import cv2
import json
import argparse
from scipy.optimize import minimize
from pathlib import Path
import sys

# --- 部分 1: 内置数据和参考颜色处理函数 ---

def get_reference_lab_data():
    """
    返回内置的标准ColorChecker 24色块Lab数据 (D50, 2度观察者)。
    数据来源于ColorChecker护照(2014年后版本)的社区测量平均值。
    """
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
    # D50白点 (CIE 1931, 2度观察者)
    Xn, Yn, Zn = 0.96422, 1.00000, 0.82521
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(f):
        f_cubed = f ** 3
        return f_cubed if f_cubed > 0.008856 else (f - 16/116) / 7.787
    
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    
    return [X, Y, Z]

def chromatic_adaptation(xyz_d50):
    """使用Bradford变换将XYZ从D50调整到D60 (ACEScg的白点)。"""
    # Bradford变换矩阵
    M_A = np.array([
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296]
    ])
    M_A_inv = np.linalg.inv(M_A)

    # D50 和 D60 白点
    XYZ_D50 = np.array([0.96422, 1.00000, 0.82521])
    XYZ_D60 = np.array([0.95255, 1.00000, 1.00882]) # ACEScg AP0 whitepoint

    # 计算cone响应
    rgb_cone_d50 = M_A @ XYZ_D50
    rgb_cone_d60 = M_A @ XYZ_D60
    
    # 转换矩阵
    adaptation_matrix = M_A_inv @ np.diag(rgb_cone_d60 / rgb_cone_d50) @ M_A
    
    # 应用变换
    xyz_d60 = adaptation_matrix @ np.array(xyz_d50)
    return xyz_d60.tolist()

def xyz_to_acescg_rgb(xyz):
    """将XYZ值(D60)转换为ACEScg RGB值。"""
    # 此矩阵为官方XYZ to ACEScg (AP1)的转换矩阵
    xyz_to_ap1_matrix = np.array([
        [ 1.64102338, -0.32480335, -0.23642469],
        [-0.66366286,  1.6153316,   0.01675635],
        [ 0.0117219,  -0.00828444,  0.98839486]
    ])
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
        # 参考密度为负值
        density = np.log10(np.clip(rgb, 1e-10, 1.0)).tolist()
        reference_densities[patch_id] = density
    print("✓ Reference densities calculated for 24 patches.")
    return reference_densities


# --- 部分 2: 扫描图像处理函数 ---

COLORCHECKER_LAYOUT = {
    'A1': (0, 0), 'A2': (0, 1), 'A3': (0, 2), 'A4': (0, 3), 'A5': (0, 4), 'A6': (0, 5),
    'B1': (1, 0), 'B2': (1, 1), 'B3': (1, 2), 'B4': (1, 3), 'B5': (1, 4), 'B6': (1, 5),
    'C1': (2, 0), 'C2': (2, 1), 'C3': (2, 2), 'C4': (2, 3), 'C5': (2, 4), 'C6': (2, 5),
    'D1': (3, 0), 'D2': (3, 1), 'D3': (3, 2), 'D4': (3, 3), 'D5': (3, 4), 'D6': (3, 5)
}

def load_image(image_path):
    """加载图像并将其转换为[0, 1]范围的float32 RGB格式。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image.dtype != np.float32:
        if np.max(image) > 255: # 假设为16位
            image = image.astype(np.float32) / 65535.0
        else: # 假设为8位
            image = image.astype(np.float32) / 255.0
            
    return np.clip(image, 0.0, 1.0)

def extract_patches_from_image(image, patch_size_ratio=0.3):
    """从图像中提取24个色块的平均RGB值。"""
    height, width = image.shape[:2]
    patch_height = height // 4
    patch_width = width // 6
    center_height = int(patch_height * patch_size_ratio)
    center_width = int(patch_width * patch_size_ratio)
    
    patches = {}
    for patch_id, (row, col) in COLORCHECKER_LAYOUT.items():
        y_start = row * patch_height
        x_start = col * patch_width
        center_y_start = y_start + (patch_height - center_height) // 2
        center_x_start = x_start + (patch_width - center_width) // 2
        
        center_patch = image[
            center_y_start : center_y_start + center_height,
            center_x_start : center_x_start + center_width
        ]
        patches[patch_id] = np.mean(center_patch, axis=(0, 1)).tolist()
        
    return patches

def process_image_data(image_path):
    """加载图像，提取色块，并计算密度。"""
    print(f"\nStep 2: Processing image '{Path(image_path).name}'...")
    image = load_image(image_path)
    patch_rgbs = extract_patches_from_image(image)
    
    image_densities = {}
    for patch_id, rgb in patch_rgbs.items():
        # 图像密度为正值
        density = (-np.log10(np.clip(rgb, 1e-10, 1.0))).tolist()
        image_densities[patch_id] = density
    print("✓ Image densities calculated for 24 patches.")
    return image_densities


# --- 部分 3: 矩阵拟合函数 ---

def prepare_data_for_fitting(ref_densities, img_densities, gray_patch_weight=20.0, skin_tone_weight=20.0):
    """准备用于优化的Numpy数组和权重。"""
    print("\nStep 3: Preparing data for fitting...")
    D_ref_list, D_img_list, weights_list = [], [], []
    
    common_patches = sorted(list(set(ref_densities.keys()) & set(img_densities.keys())))
    if len(common_patches) != 24:
        print(f"Warning: Only {len(common_patches)} matching patches found. Expected 24.")

    for patch_id in common_patches:
        D_ref_list.append(ref_densities[patch_id])
        D_img_list.append(img_densities[patch_id])
        
        # 为特定色块分配权重
        if patch_id in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']: # D1-D6 灰阶
            weights_list.append(gray_patch_weight)
        elif patch_id in ['A1', 'A2']: # A1, A2 肤色
            weights_list.append(skin_tone_weight)
        else:
            weights_list.append(1.0)
            
    D_ref = np.array(D_ref_list)
    D_img = np.array(D_img_list)
    weights = np.array(weights_list).reshape(-1, 1)
    
    print(f"✓ Data prepared. Found {len(common_patches)} patch pairs.")
    print(f"  - Gray-scale patches (D1-D6) weight: {gray_patch_weight}x")
    print(f"  - Skin-tone patches (A1, A2) weight: {skin_tone_weight}x")
    print(f"  Reference density range: [{D_ref.min():.4f}, {D_ref.max():.4f}]")
    print(f"  Image density range:     [{D_img.min():.4f}, {D_img.max():.4f}]")
    return D_ref, D_img, weights

def objective_function(params, D_ref, D_img, weights):
    """优化器使用的目标函数(加权均方误差)。模型: D_ref^T = M @ D_img^T + g^T"""
    M = params[:9].reshape(3, 3)
    g = params[9:12]
    
    # D_img shape: (24, 3). D_img.T shape: (3, 24)
    # M @ D_img.T shape: (3, 24)
    # D_pred.T shape: (3, 24) -> D_pred shape: (24, 3)
    D_pred = (M @ D_img.T + g.reshape(-1, 1)).T
    
    # 计算加权误差
    error = (D_ref - D_pred) ** 2
    weighted_error = error * weights
    
    return np.mean(weighted_error)

def fit_matrix(D_ref, D_img, weights):
    """执行拟合操作以找到矩阵M和向量g。"""
    print("\nStep 4: Fitting density relationship matrix...")
    # 初始猜测：M为-1单位矩阵，g为零向量
    initial_params = np.array([-1., 0., 0., 0., -1., 0., 0., 0., -1., 0., 0., 0.])
    
    result = minimize(
        objective_function,
        initial_params,
        args=(D_ref, D_img, weights),
        method='L-BFGS-B',
        options={'maxiter': 10000}
    )
    
    if not result.success:
        print(f"Warning: Optimization may not have converged. Message: {result.message}")
        
    M = result.x[:9].reshape(3, 3)
    g = result.x[9:12]
    mse = result.fun
    
    print(f"✓ Fitting complete. Final Mean Squared Error: {mse:.6f}")
    return M, g, mse

def normalize_matrix(M):
    """根据M(1,1)标准化矩阵，使得左上角元素为1。"""
    M11 = M[0, 0]
    if abs(M11) < 1e-10:
        print("Warning: M(1,1) is very close to zero, normalization may be unstable.")
        return M, M11
    
    M_normalized = M / M11
    return M_normalized, M11


def apply_and_save_correction(image_path, M, g):
    """应用校正矩阵并保存结果图像。"""
    print("\nStep 7: Applying correction to the image and saving...")
    try:
        # 1. 加载原始图像
        image_rgb_float = load_image(image_path)
        
        # 2. 将图像从RGB转换到密度空间
        # 我们需要处理0值，避免log(0)错误
        image_density = -np.log10(np.clip(image_rgb_float, 1e-10, 1.0))
        
        # 3. 应用校正
        # D_corrected^T = M @ D_image^T + g^T
        # Reshape image_density to (height*width, 3) for matrix multiplication
        height, width, _ = image_density.shape
        image_density_flat = image_density.reshape(-1, 3) # Shape: (N, 3)
        
        # Transpose for M @ D calculation
        image_density_flat_T = image_density_flat.T # Shape: (3, N)
        corrected_density_flat_T = M @ image_density_flat_T + g.reshape(-1, 1) # Shape: (3, N)
        
        # Transpose back and reshape
        corrected_density_flat = corrected_density_flat_T.T # Shape: (N, 3)
        corrected_density = corrected_density_flat.reshape(height, width, 3)
        
        # 4. 将校正后的密度转换回RGB空间
        # RGB = 10^(-D)
        corrected_rgb_float = 10**(corrected_density)
        
        # 5. 裁剪并转换为16位整数格式
        corrected_rgb_clipped = np.clip(corrected_rgb_float, 0.0, 1.0)
        corrected_rgb_16bit = (corrected_rgb_clipped * 65535).astype(np.uint16)

        # 6. 转换回BGR以供OpenCV保存
        corrected_bgr_16bit = cv2.cvtColor(corrected_rgb_16bit, cv2.COLOR_RGB2BGR)

        # 7. 生成输出文件名并保存
        p = Path(image_path)
        output_path = p.with_name(f"{p.stem}_corrected{p.suffix}")
        cv2.imwrite(str(output_path), corrected_bgr_16bit)
        
        print(f"✓ Corrected image saved successfully to:\n{output_path.resolve()}")

    except Exception as e:
        print(f"\n⚠️ Could not save corrected image. Error: {e}", file=sys.stderr)



# --- 部分 4: 主函数 ---

def main():
    """主执行函数。"""
    parser = argparse.ArgumentParser(
        description='从ColorChecker照片中计算扫描仪校准矩阵。',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "用法示例:\n"
            "  python calibrate_scanner.py ./test_scans/CC0-Croped.tif ./config/matrices/my_scanner_matrix.json"
        )
    )
    parser.add_argument('input_image', help='输入的ColorChecker图像文件路径 (应为4x6的已裁切图像)。')
    parser.add_argument('output_json', help='输出的校准矩阵JSON文件路径。')
    parser.add_argument('--save-corrected-image', action='store_true', help='应用校正并保存一个新的图像文件。')
    
    args = parser.parse_args()
    
    try:
        # 1. 处理参考数据
        ref_densities = process_reference_data()
        
        # 2. 处理图像数据
        img_densities = process_image_data(args.input_image)
        
        # 3. 准备数据
        D_ref, D_img, weights = prepare_data_for_fitting(ref_densities, img_densities)
        
        # 4. 拟合矩阵
        M, g, mse = fit_matrix(D_ref, D_img, weights)
        
        # 5. 标准化矩阵
        print("\nStep 5: Normalizing matrix...")
        M_normalized, M11 = normalize_matrix(M)
        print(f"✓ Matrix normalized. M(1,1) = {M11:.6f}")
        
        # 6. 保存结果
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'description': 'Scanner calibration matrix and offset vector.',
            'relationship': 'D_reference^T = M @ D_image^T + g^T',
            'M_matrix': M.tolist(),
            'M_normalized': M_normalized.tolist(),
            'M11_value': M11,
            'g_vector': g.tolist(),
            'mean_squared_error': mse
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Calibration successful! Results saved to:\n{output_path.resolve()}")
        
        print("\n--- Final Matrix M ---")
        for row in M:
            print(f"  [{row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f}]")
        print("\n--- Normalized Matrix M' ---")
        for row in M_normalized:
            print(f"  [{row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>8.4f}]")
        print(f"\n--- M(1,1) Value ---")
        print(f"  {M11:.6f}")
        print("\n--- Final Vector g ---")
        print(f"  [{g[0]:>8.4f} {g[1]:>8.4f} {g[2]:>8.4f}]")

        # 7. 如果用户要求，应用校正并保存图像
        if args.save_corrected_image:
            apply_and_save_correction(args.input_image, M, g)


    except Exception as e:
        print(f"\n❌ An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

