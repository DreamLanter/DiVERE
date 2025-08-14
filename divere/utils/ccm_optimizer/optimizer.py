#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCM参数优化器

使用 CMA-ES 优化 DiVERE 管线的关键参数：
- primaries_xy: 输入色彩空间的RGB基色
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
                 weights_config_path: Optional[str] = None):
        """
        初始化优化器
        
        Args:
            reference_file: 参考RGB值文件路径
        """
        self.pipeline = DiVEREPipelineSimulator(verbose=False)
        self.reference_values = self._load_reference_values(reference_file)
        # 加载权重配置（可选）。默认使用内置的 config/colorchecker/weights.json
        self._weights_config = self._load_weights_config(weights_config_path)
        self._patch_weight_map = self._build_patch_weight_map(self._weights_config)
        self._channel_weights = self._load_channel_weights(self._weights_config)
        
        # 参数边界和初始值
        self.bounds = {
            'primaries_xy': [
                # R基色 x, y
                (0.0, 1.0), (0.0, 1.0),  # R基色范围
                # G基色 x, y  
                (0.0, 1.0), (0.0, 1.0),  # G基色范围
                # B基色 x, y
                (0.0, 1.0), (0.0, 1.0)   # B基色范围
            ],
            'gamma': (1.0, 4.0),          # 密度反差范围
            'dmax': (0.5, 4.0),           # 最大密度范围
            'r_gain': (-2, 2),        # R增益范围
            'b_gain': (-2, 2)         # B增益范围
        }
        
        self.initial_params = {
            'primaries_xy': np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06]),  # sRGB基色
            'gamma': 2.6,      # DiVERE默认值
            'dmax': 2.0,       # DiVERE默认值
            'r_gain': 0.0,     # 无增益
            'b_gain': 0.0      # 无增益
        }
    
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
        """
        mapping: Dict[str, float] = {}
        try:
            w = cfg.get("weights", {}) or {}
            cats = cfg.get("patch_categories", {}) or {}
            for cat, ids in cats.items():
                weight_key = f"{cat}_weight"
                wv = float(w.get(weight_key, 1.0))
                for pid in ids or []:
                    mapping[str(pid)] = wv
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
        """将优化参数数组转换为参数字典"""
        return {
            'primaries_xy': params[:6].reshape(3, 2),
            'gamma': params[6],
            'dmax': params[7], 
            'r_gain': params[8],
            'b_gain': params[9]
        }
    
    def _dict_to_params(self, params_dict: Dict) -> np.ndarray:
        """将参数字典转换为优化参数数组"""
        return np.concatenate([
            params_dict['primaries_xy'].flatten(),
            [params_dict['gamma']],
            [params_dict['dmax']],
            [params_dict['r_gain']],
            [params_dict['b_gain']]
        ])
    
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
            output_patches = self.pipeline.simulate_full_pipeline(
                input_patches,
                primaries_xy=params_dict['primaries_xy'],
                gamma=params_dict['gamma'],
                dmax=params_dict['dmax'],
                r_gain=params_dict['r_gain'],
                b_gain=params_dict['b_gain'],
                correction_matrix=correction_matrix,
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
                 max_iter: int = 300,
                 tolerance: float = 1e-8,
                 correction_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        执行优化
        
        Args:
            input_patches: 输入色块RGB值
            method: 优化方法
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            优化结果字典
        """
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
        # 运行管线
        output_patches = self.pipeline.simulate_full_pipeline(
            input_patches,
            primaries_xy=params_dict['primaries_xy'],
            gamma=params_dict['gamma'],
            dmax=params_dict['dmax'],
            r_gain=params_dict['r_gain'],
            b_gain=params_dict['b_gain'],
            correction_matrix=correction_matrix,
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
        """生成 (lb, ub, span) 数组，按参数展平顺序对应: rx, ry, gx, gy, bx, by, gamma, dmax, r_gain, b_gain"""
        prim_bounds = self.bounds['primaries_xy']  # list of 6 tuples
        lb_list = [
            prim_bounds[0][0], prim_bounds[1][0],  # rx, ry
            prim_bounds[2][0], prim_bounds[3][0],  # gx, gy
            prim_bounds[4][0], prim_bounds[5][0],  # bx, by
            self.bounds['gamma'][0],
            self.bounds['dmax'][0],
            self.bounds['r_gain'][0],
            self.bounds['b_gain'][0],
        ]
        ub_list = [
            prim_bounds[0][1], prim_bounds[1][1],  # rx, ry
            prim_bounds[2][1], prim_bounds[3][1],  # gx, gy
            prim_bounds[4][1], prim_bounds[5][1],  # bx, by
            self.bounds['gamma'][1],
            self.bounds['dmax'][1],
            self.bounds['r_gain'][1],
            self.bounds['b_gain'][1],
        ]
        lb = np.array(lb_list, dtype=float)
        ub = np.array(ub_list, dtype=float)
        span = np.maximum(ub - lb, 1e-6)
        return lb, ub, span

    def _optimize_cma(self,
                      input_patches: Dict[str, Tuple[float, float, float]],
                      max_iter: int = 300,
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
