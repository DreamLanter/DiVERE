#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCM参数优化器

使用 CMA-ES 优化 DiVERE 管线的关键参数：
- primaries_xy: 输入色彩变换的RGB基色
- gamma: 密度反差参数
- dmax: 最大密度参数
- r_gain: R通道增益
- b_gain: B通道增益

目标：最小化24个 ColorChecker 色块的 RGB RMSE
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import sys

# 项目根目录（源码情形下可用），冻结环境下可能无效，但不强依赖
project_root = Path(__file__).parent.parent.parent

# 兼容导入（冻结/源码/包内）
try:
    from divere.utils.ccm_optimizer.pipeline import DiVEREPipelineSimulator  # type: ignore
    from divere.utils.ccm_optimizer.extractor import extract_colorchecker_patches  # type: ignore
except Exception:
    try:
        from .pipeline import DiVEREPipelineSimulator  # type: ignore
        from .extractor import extract_colorchecker_patches  # type: ignore
    except Exception:
        try:
            from utils.ccm_optimizer.pipeline import DiVEREPipelineSimulator  # type: ignore
            from utils.ccm_optimizer.extractor import extract_colorchecker_patches  # type: ignore
        except Exception as e:
            raise ImportError(f"无法导入CCM优化器依赖: {e}")

class CCMOptimizer:
    """CCM参数优化器"""
    
    def __init__(self, reference_file: str = "colorchecker_acescg_rgb_values.json",
                 weights_config_path: Optional[str] = None,
                 sharpening_config: Optional['SpectralSharpeningConfig'] = None):
        """
        初始化优化器
        
        Args:
            reference_file: 参考RGB值文件路径
            weights_config_path: 权重配置文件路径
            sharpening_config: 光谱锐化配置，决定哪些参数参与优化
        """
        # 导入配置类（避免循环导入）
        if sharpening_config is None:
            try:
                from divere.core.data_types import SpectralSharpeningConfig
                sharpening_config = SpectralSharpeningConfig()
            except ImportError:
                # 向后兼容：创建一个简单的配置对象
                class DefaultConfig:
                    optimize_idt_transformation = True
                    optimize_density_matrix = False
                sharpening_config = DefaultConfig()
        
        self.sharpening_config = sharpening_config
        self.pipeline = DiVEREPipelineSimulator(verbose=False)
        self.reference_values = self._load_reference_values(reference_file)
        # 加载权重配置（可选）。默认使用内置的 config/colorchecker/weights.json
        self._weights_config = self._load_weights_config(weights_config_path)
        self._patch_weight_map = self._build_patch_weight_map(self._weights_config)
        self._channel_weights = self._load_channel_weights(self._weights_config)
        
        # 根据配置构建参数映射
        self._build_parameter_mapping()
        
    def _build_parameter_mapping(self):
        """根据配置动态构建参数边界、初始值和索引映射"""
        self.bounds = {}
        self.initial_params = {}
        self._param_indices = {}  # 参数名到数组索引的映射
        
        current_idx = 0
        
        # IDT transformation参数（始终包含gamma, dmax, r_gain, b_gain）
        self.bounds['gamma'] = (1.0, 4.0)
        self.bounds['dmax'] = (0.5, 4.0)  
        self.bounds['r_gain'] = (-2, 2)
        self.bounds['b_gain'] = (-2, 2)
        
        self.initial_params['gamma'] = 2.0
        self.initial_params['dmax'] = 2.0
        self.initial_params['r_gain'] = 0.0
        self.initial_params['b_gain'] = 0.0
        
        self._param_indices['gamma'] = current_idx
        self._param_indices['dmax'] = current_idx + 1
        self._param_indices['r_gain'] = current_idx + 2  
        self._param_indices['b_gain'] = current_idx + 3
        current_idx += 4
        
        # primaries_xy（如果启用IDT优化）
        if self.sharpening_config.optimize_idt_transformation:
            self.bounds['primaries_xy'] = [
                # R基色 x, y
                (0.0, 1.0), (0.0, 1.0),  # R基色范围
                # G基色 x, y  
                (0.0, 1.0), (0.0, 1.0),  # G基色范围
                # B基色 x, y
                (0.0, 1.0), (0.0, 1.0)   # B基色范围
            ]
            self.initial_params['primaries_xy'] = np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06])  # sRGB基色
            self._param_indices['primaries_xy'] = slice(current_idx, current_idx + 6)
            current_idx += 6
        
        # density_matrix（如果启用density matrix优化）  
        if self.sharpening_config.optimize_density_matrix:
            # 8个参数：3x3矩阵除了左上角(0,0)固定为1.0的其余8个元素
            self.bounds['density_matrix'] = [
                # 跳过(0,0)，从(0,1)开始：第一行剩余2个
                (-0.5, 0.5), (-0.5, 0.5),
                # 第二行3个
                (-0.5, 0.5), (0.5, 2.0), (-0.5, 0.5),
                # 第三行3个
                (-0.5, 0.5), (-0.5, 0.5), (0.5, 2.0)
            ]
            # 初始为单位矩阵的8个可变元素（跳过左上角）
            self.initial_params['density_matrix'] = np.array([
                0.0, 0.0,       # 第一行: (0,1), (0,2)
                0.0, 1.0, 0.0,  # 第二行: (1,0), (1,1), (1,2)
                0.0, 0.0, 1.0   # 第三行: (2,0), (2,1), (2,2)
            ])
            self._param_indices['density_matrix'] = slice(current_idx, current_idx + 8)
            current_idx += 8
        
        self._total_params = current_idx
    
    def _update_initial_params_from_ui(self, ui_params: Dict):
        """根据UI当前参数更新优化初值"""
        # 更新基础参数
        if 'gamma' in ui_params:
            self.initial_params['gamma'] = float(ui_params['gamma'])
        if 'dmax' in ui_params:
            self.initial_params['dmax'] = float(ui_params['dmax'])
        if 'r_gain' in ui_params:
            self.initial_params['r_gain'] = float(ui_params['r_gain'])
        if 'b_gain' in ui_params:
            self.initial_params['b_gain'] = float(ui_params['b_gain'])
            
        # 更新primaries_xy（如果启用优化）
        if 'primaries_xy' in self._param_indices and 'primaries_xy' in ui_params:
            self.initial_params['primaries_xy'] = np.array(ui_params['primaries_xy']).flatten()
            
        # 更新density_matrix（如果启用优化）
        if 'density_matrix' in self._param_indices and 'density_matrix' in ui_params:
            matrix = np.array(ui_params['density_matrix'])
            # 提取8个可变元素作为初值（左上角固定为1.0）
            self.initial_params['density_matrix'] = np.array([
                matrix[0, 1], matrix[0, 2],  # 第一行剩余2个
                matrix[1, 0], matrix[1, 1], matrix[1, 2],  # 第二行3个
                matrix[2, 0], matrix[2, 1], matrix[2, 2]   # 第三行3个
            ])
    
    def _load_reference_values(self, reference_file: str) -> Dict[str, List[float]]:
        """加载参考RGB值
        优先从统一数据路径 `config/colorchecker/<file>` 解析；
        冻结环境下优先使用可执行目录旁的 `config`；开发环境回退到包内同目录或 `divere/config/colorchecker`。
        """
        candidates: List[Path] = []
        # 允许外部传绝对路径
        rf = Path(reference_file)
        if rf.is_absolute():
            candidates.append(rf)
        else:
            # 1) 统一资源定位：config/colorchecker/<name>
            try:
                from divere.utils.app_paths import resolve_data_path  # type: ignore
                candidates.append(resolve_data_path("config", "colorchecker", reference_file))
            except Exception:
                pass
            # 2) 包内同目录（兼容旧路径）
            candidates.append(Path(__file__).parent / reference_file)
            # 3) 包内标准位置：divere/config/colorchecker/<name>
            candidates.append(project_root / "config" / "colorchecker" / reference_file)

        for p in candidates:
            try:
                if p.exists():
                    with open(p, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        return data.get('data', {})
            except Exception:
                # 继续尝试下一个候选
                pass

        print(
            f"警告：无法加载参考文件 {reference_file}: 未在以下位置找到可用文件: "
            f"{[str(x) for x in candidates]}"
        )
        return {}

    # ===== 权重：加载与查询 =====
    def _load_weights_config(self, weights_config_path: Optional[str]) -> Dict[str, Any]:
        """加载色块权重配置；失败时返回默认等权。"""
        # 默认路径：使用统一的资源解析入口
        if weights_config_path:
            path = Path(weights_config_path)
        else:
            try:
                from divere.utils.app_paths import resolve_data_path
                path = resolve_data_path("config", "colorchecker", "weights.json")
            except Exception:
                path = project_root / "divere" / "config" / "colorchecker" / "weights.json"
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
        except Exception as e:
            print(f"警告：无法加载权重配置 {path}: {e}")
        # 回退：等权配置
        return {
            "weights": {"grayscale_weight": 1.0, "skin_weight": 1.0, "color_weight": 1.0},
            "patch_categories": {
                "grayscale": [], "skin": [], "color": []
            }
        }

    def _build_patch_weight_map(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        """根据类别配置生成每个色块到权重的映射。
        通用化：对 cfg['patch_categories'] 中的每个类别 cat，优先查找
        cfg['weights'][f"{cat}_weight"] 作为该类别权重（缺省=1）。
        支持自定义类别，例如 'primaries' + 'primaries_weight'。
        支持individual_weights覆盖特定色块权重。
        """
        mapping: Dict[str, float] = {}
        try:
            w = cfg.get("weights", {}) or {}
            cats = cfg.get("patch_categories", {}) or {}
            # 首先根据类别设置权重
            for cat, ids in cats.items():
                weight_key = f"{cat}_weight"
                wv = float(w.get(weight_key, 1.0))
                for pid in ids or []:
                    mapping[str(pid)] = wv
            
            # 然后应用个别权重覆盖（优先级更高）
            individual_weights = cfg.get("individual_weights", {}) or {}
            for patch_id, weight in individual_weights.items():
                mapping[str(patch_id)] = float(weight)
        except Exception:
            pass
        return mapping

    def _get_patch_weight(self, patch_id: str) -> float:
        return float(self._patch_weight_map.get(patch_id, 1.0))

    def _load_channel_weights(self, cfg: Dict[str, Any]) -> np.ndarray:
        """加载每通道主色权重（R/G/B）。缺省为[1,1,1]。支持
        cfg['primary_weight'] = {"R": w_r, "G": w_g, "B": w_b}
        """
        try:
            p = cfg.get('primary_weight', {}) or {}
            # 兼容大小写与数组形式
            if isinstance(p, dict):
                wr = float(p.get('R', p.get('r', 1.0)))
                wg = float(p.get('G', p.get('g', 1.0)))
                wb = float(p.get('B', p.get('b', 1.0)))
                v = np.array([wr, wg, wb], dtype=float)
            elif isinstance(p, (list, tuple)) and len(p) == 3:
                v = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=float)
            else:
                v = np.array([1.0, 1.0, 1.0], dtype=float)
        except Exception:
            v = np.array([1.0, 1.0, 1.0], dtype=float)
        # 避免全零
        if not np.isfinite(v).all() or float(v.sum()) <= 0.0:
            v = np.array([1.0, 1.0, 1.0], dtype=float)
        return v
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """将优化参数数组转换为参数字典（动态映射版本）"""
        result = {}
        
        # 基础参数（始终存在）
        result['gamma'] = params[self._param_indices['gamma']]
        result['dmax'] = params[self._param_indices['dmax']]
        result['r_gain'] = params[self._param_indices['r_gain']]
        result['b_gain'] = params[self._param_indices['b_gain']]
        
        # primaries_xy（如果启用IDT优化）
        if 'primaries_xy' in self._param_indices:
            idx_slice = self._param_indices['primaries_xy']
            result['primaries_xy'] = params[idx_slice].reshape(3, 2)
        else:
            # 使用sRGB基色作为固定值（当不优化IDT时）
            result['primaries_xy'] = np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06]).reshape(3, 2)
        
        # density_matrix（如果启用优化）
        if 'density_matrix' in self._param_indices:
            idx_slice = self._param_indices['density_matrix']
            # 重建3x3矩阵：左上角固定为1.0，其余8个元素来自优化参数
            matrix_params = params[idx_slice]  # 8个参数
            matrix = np.zeros((3, 3), dtype=float)
            matrix[0, 0] = 1.0  # 固定左上角
            matrix[0, 1] = matrix_params[0]  # (0,1)
            matrix[0, 2] = matrix_params[1]  # (0,2)
            matrix[1, 0] = matrix_params[2]  # (1,0)
            matrix[1, 1] = matrix_params[3]  # (1,1)
            matrix[1, 2] = matrix_params[4]  # (1,2)
            matrix[2, 0] = matrix_params[5]  # (2,0)
            matrix[2, 1] = matrix_params[6]  # (2,1)
            matrix[2, 2] = matrix_params[7]  # (2,2)
            result['density_matrix'] = matrix
        else:
            result['density_matrix'] = None
        
        return result
    
    def _dict_to_params(self, params_dict: Dict) -> np.ndarray:
        """将参数字典转换为优化参数数组（动态映射版本）"""
        params = np.zeros(self._total_params, dtype=float)
        
        # 基础参数（始终存在）
        params[self._param_indices['gamma']] = params_dict['gamma']
        params[self._param_indices['dmax']] = params_dict['dmax']
        params[self._param_indices['r_gain']] = params_dict['r_gain'] 
        params[self._param_indices['b_gain']] = params_dict['b_gain']
        
        # primaries_xy（如果启用优化）
        if 'primaries_xy' in self._param_indices:
            idx_slice = self._param_indices['primaries_xy']
            if 'primaries_xy' in params_dict:
                params[idx_slice] = params_dict['primaries_xy'].flatten()
            
        # density_matrix（如果启用优化）
        if 'density_matrix' in self._param_indices:
            idx_slice = self._param_indices['density_matrix']
            if 'density_matrix' in params_dict and params_dict['density_matrix'] is not None:
                matrix = params_dict['density_matrix']
                # 如果matrix已经是1D数组（从initial_params来），直接使用
                if matrix.ndim == 1 and len(matrix) == 8:
                    params[idx_slice] = matrix
                else:
                    # 如果是3x3矩阵，提取8个可变元素（跳过左上角(0,0)）
                    matrix_params = np.array([
                        matrix[0, 1], matrix[0, 2],  # 第一行剩余2个
                        matrix[1, 0], matrix[1, 1], matrix[1, 2],  # 第二行3个
                        matrix[2, 0], matrix[2, 1], matrix[2, 2]   # 第三行3个
                    ])
                    params[idx_slice] = matrix_params
            
        return params
    
    def objective_function(self, params: np.ndarray, 
                          input_patches: Dict[str, Tuple[float, float, float]]) -> float:
        """
        目标函数：计算RGB RMSE
        
        Args:
            params: 优化参数数组
            input_patches: 输入色块RGB值
            
        Returns:
            MSE值
        """
        # 兼容旧签名：默认不使用校正矩阵
        return self.compute_rmse(params, input_patches, correction_matrix=None)

    def compute_rmse(self, params: np.ndarray,
                     input_patches: Dict[str, Tuple[float, float, float]],
                     correction_matrix: Optional[np.ndarray] = None) -> float:
        """计算给定参数与可选密度校正矩阵下的全局RMSE。"""
        try:
            params_dict = self._params_to_dict(params)
            
            # 确定实际使用的correction_matrix
            actual_correction_matrix = correction_matrix
            if params_dict['density_matrix'] is not None:
                # 如果参数中包含优化的density_matrix，优先使用它
                actual_correction_matrix = params_dict['density_matrix']
            
            output_patches = self.pipeline.simulate_full_pipeline(
                input_patches,
                primaries_xy=params_dict['primaries_xy'],
                gamma=params_dict['gamma'],
                dmax=params_dict['dmax'],
                r_gain=params_dict['r_gain'],
                b_gain=params_dict['b_gain'],
                correction_matrix=actual_correction_matrix,
            )
            total_weighted_mse = 0.0
            total_weight = 0.0
            for patch_id in self.reference_values.keys():
                if patch_id in output_patches:
                    ref_rgb = np.array(self.reference_values[patch_id])
                    out_rgb = np.array(output_patches[patch_id])
                    delta = ref_rgb - out_rgb
                    # 每通道主色权重
                    wr, wg, wb = self._channel_weights
                    ch_denom = float(wr + wg + wb)
                    if ch_denom > 0.0:
                        patch_mse = float((wr * delta[0]**2 + wg * delta[1]**2 + wb * delta[2]**2) / ch_denom)
                    else:
                        patch_mse = float(np.mean(delta**2))
                    w = self._get_patch_weight(patch_id)
                    total_weighted_mse += float(w) * float(patch_mse)
                    total_weight += float(w)
            if total_weight <= 0.0:
                return float('inf')
            return float(np.sqrt(total_weighted_mse / total_weight))
        except Exception as e:
            print(f"目标函数计算错误: {e}")
            return float('inf')
    
    def optimize(self, input_patches: Dict[str, Tuple[float, float, float]],
                 method: str = 'CMA-ES',
                 max_iter: int = 1000,
                 tolerance: float = 1e-8,
                 correction_matrix: Optional[np.ndarray] = None,
                 ui_params: Optional[Dict] = None) -> Dict:
        """
        执行优化
        
        Args:
            input_patches: 输入色块RGB值
            method: 优化方法
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            correction_matrix: 密度校正矩阵（如果不优化density matrix时使用）
            ui_params: 来自UI的当前参数，用作优化初值
            
        Returns:
            优化结果字典
        """
        # 如果提供了UI参数，使用它们作为初值
        if ui_params:
            print(f"使用UI参数作为初值: {ui_params}")
            self._update_initial_params_from_ui(ui_params)
            print(f"更新后的初始参数: {self.initial_params}")
        else:
            print("使用默认初始参数（未提供UI参数）")
            
        print(f"开始优化，目标：最小化24个色块的RGB RMSE")
        print(f"优化方法: {method}")
        print(f"最大迭代: {max_iter}")
        print(f"收敛容差: {tolerance}")
        
        # 统一使用 CMA-ES
        return self._optimize_cma(input_patches, max_iter=max_iter, tolerance=tolerance, correction_matrix=correction_matrix)
    
    def evaluate_parameters(self, params_dict: Dict,
                           input_patches: Dict[str, Tuple[float, float, float]],
                           correction_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        评估给定参数的性能
        
        Args:
            params_dict: 参数字典
            input_patches: 输入色块RGB值
            
        Returns:
            评估结果字典
        """
        # 确定实际使用的correction_matrix
        actual_correction_matrix = correction_matrix
        if 'density_matrix' in params_dict and params_dict['density_matrix'] is not None:
            # 如果参数中包含优化的density_matrix，优先使用它
            actual_correction_matrix = params_dict['density_matrix']
        
        # 运行管线
        output_patches = self.pipeline.simulate_full_pipeline(
            input_patches,
            primaries_xy=params_dict['primaries_xy'],
            gamma=params_dict['gamma'],
            dmax=params_dict['dmax'],
            r_gain=params_dict['r_gain'],
            b_gain=params_dict['b_gain'],
            correction_matrix=actual_correction_matrix,
        )
        
        # 计算每个色块的误差
        patch_errors: Dict[str, Any] = {}
        total_mse = 0.0
        total_weighted_mse = 0.0
        total_weight = 0.0
        valid_patches = 0
        
        for patch_id in self.reference_values.keys():
            if patch_id in output_patches:
                ref_rgb = np.array(self.reference_values[patch_id])
                out_rgb = np.array(output_patches[patch_id])
                
                # 计算误差
                error = ref_rgb - out_rgb
                # 通道加权MSE
                wr, wg, wb = self._channel_weights
                ch_denom = float(wr + wg + wb)
                if ch_denom > 0.0:
                    mse = float((wr * error[0]**2 + wg * error[1]**2 + wb * error[2]**2) / ch_denom)
                else:
                    mse = float(np.mean(error ** 2))
                rmse = float(np.sqrt(mse))
                
                w = self._get_patch_weight(patch_id)
                patch_errors[patch_id] = {
                    'reference': ref_rgb.tolist(),
                    'output': out_rgb.tolist(),
                    'error': error.tolist(),
                    'mse': float(mse),
                    'rmse': rmse,
                    'weight': float(w)
                }
                
                total_mse += mse
                total_weighted_mse += float(w) * float(mse)
                total_weight += float(w)
                valid_patches += 1
        
        avg_mse = total_mse / valid_patches if valid_patches > 0 else float('inf')
        avg_rmse = float(np.sqrt(avg_mse)) if np.isfinite(avg_mse) else float('inf')
        weighted_avg_mse = (total_weighted_mse / total_weight) if total_weight > 0 else float('inf')
        weighted_avg_rmse = float(np.sqrt(weighted_avg_mse)) if np.isfinite(weighted_avg_mse) else float('inf')
        
        return {
            'average_mse': avg_mse,
            'average_rmse': avg_rmse,
            'weighted_average_mse': weighted_avg_mse,
            'weighted_average_rmse': weighted_avg_rmse,
            'patch_errors': patch_errors,
            'valid_patches': valid_patches
        }

    # ===== CMA-ES 实现 =====
    def _build_bounds_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成 (lb, ub, span) 数组，按动态参数映射顺序"""
        lb_list = []
        ub_list = []
        
        # 按照 _build_parameter_mapping 中的顺序构建边界
        # 基础参数：gamma, dmax, r_gain, b_gain
        lb_list.append(self.bounds['gamma'][0])
        ub_list.append(self.bounds['gamma'][1])
        lb_list.append(self.bounds['dmax'][0])
        ub_list.append(self.bounds['dmax'][1])
        lb_list.append(self.bounds['r_gain'][0])
        ub_list.append(self.bounds['r_gain'][1])
        lb_list.append(self.bounds['b_gain'][0])
        ub_list.append(self.bounds['b_gain'][1])
        
        # primaries_xy（如果启用）
        if 'primaries_xy' in self.bounds:
            prim_bounds = self.bounds['primaries_xy']
            for bound in prim_bounds:
                lb_list.append(bound[0])
                ub_list.append(bound[1])
        
        # density_matrix（如果启用）
        if 'density_matrix' in self.bounds:
            matrix_bounds = self.bounds['density_matrix']
            for bound in matrix_bounds:
                lb_list.append(bound[0])
                ub_list.append(bound[1])
        
        lb = np.array(lb_list, dtype=float)
        ub = np.array(ub_list, dtype=float)
        span = np.maximum(ub - lb, 1e-6)
        return lb, ub, span

    def _optimize_cma(self,
                      input_patches: Dict[str, Tuple[float, float, float]],
                      max_iter: int = 1000,
                      tolerance: float = 1e-8,
                      correction_matrix: Optional[np.ndarray] = None) -> Dict:
        try:
            import cma
        except Exception as e:
            raise RuntimeError(f"请先安装 cma: pip install cma ({e})")

        x0 = self._dict_to_params(self.initial_params)
        lb, ub, span = self._build_bounds_arrays()

        sigma0 = 0.15
        opts = {
            'bounds': [lb.tolist(), ub.tolist()],
            'scaling_of_variables': span.tolist(),
            'maxiter': int(max_iter),
            'ftarget': float(tolerance),
            'verb_disp': 1,
            'verbose': 1,
            'popsize': int(8 + 4 * np.log(len(x0))),
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        best_rmse = float('inf')
        while not es.stop():
            xs = es.ask()
            fs = [float(self.compute_rmse(x, input_patches, correction_matrix=correction_matrix)) for x in xs]
            es.tell(xs, fs)
            es.disp()
            gen_best = float(np.min(fs))
            if gen_best < best_rmse:
                best_rmse = gen_best
            print(f"迭代 {es.countiter:3d}: RMSE={gen_best:.6f}  (累计最优={best_rmse:.6f})")

        res = es.result  # type: ignore[attr-defined]
        xbest = np.array(res.xbest, dtype=float)
        fbest = float(res.fbest)
        nit = int(es.countiter)
        optimal_params = self._params_to_dict(xbest)
        print("✓ 优化成功完成")
        print(f"最终RMSE: {fbest:.6f}")
        print(f"迭代次数: {nit}")
        return {
            'success': True,
            'rmse': fbest,
            'iterations': nit,
            'parameters': optimal_params,
            'raw_result': res,
        }

def optimize_from_image(image_array: np.ndarray,
                       corners: List[Tuple[float, float]],
                       reference_file: str = "colorchecker_acescg_rgb_values.json",
                       **optimizer_kwargs) -> Dict:
    """
    从图像直接优化的便捷函数
    
    Args:
        image_array: 图像数组
        corners: ColorChecker四角点坐标
        reference_file: 参考文件路径
        **optimizer_kwargs: 传递给优化器的参数
        
    Returns:
        优化结果字典
    """
    # 提取色块
    print("提取ColorChecker色块...")
    input_patches = extract_colorchecker_patches(image_array, corners)
    
    if not input_patches:
        raise ValueError("无法提取色块数据")
    
    print(f"成功提取 {len(input_patches)} 个色块")
    
    # 创建优化器并执行优化
    optimizer = CCMOptimizer(reference_file)
    result = optimizer.optimize(input_patches, **optimizer_kwargs)
    
    # 评估最终结果
    evaluation = optimizer.evaluate_parameters(result['parameters'], input_patches)
    result['evaluation'] = evaluation
    
    return result

if __name__ == "__main__":
    # 测试代码
    print("CCM优化器测试")
    
    # 创建测试数据
    test_patches = {
        'A1': (0.1, 0.1, 0.1),
        'D6': (0.9, 0.9, 0.9),
        'B3': (0.4, 0.1, 0.1)
    }
    
    optimizer = CCMOptimizer()
    
    # 测试目标函数
    test_params = optimizer._dict_to_params(optimizer.initial_params)
    rmse = optimizer.objective_function(test_params, test_patches)
    print(f"初始参数RMSE: {rmse:.6f}")
    
    # 测试优化
    print("\n开始测试优化...")
    result = optimizer.optimize(test_patches, max_iter=10)
    print(f"优化结果: {result['success']}")
    print(f"最终RMSE: {result['rmse']:.6f}")
