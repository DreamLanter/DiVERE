"""
LUT处理模块
LUT生成和应用
"""

import numpy as np
from typing import Optional
import time

from .data_types import LUT3D, ImageData, ColorGradingParams
from .the_enlarger import TheEnlarger


class LUTProcessor:
    """LUT处理器"""
    
    def __init__(self, the_enlarger: TheEnlarger):
        self.the_enlarger = the_enlarger
        self._lut_cache = {}
        self._max_cache_size = 20
    
    def generate_preview_lut(self, params: ColorGradingParams, size: int = 32) -> LUT3D:
        """生成预览LUT"""
        # 检查缓存
        cache_key = self._get_params_hash(params)
        if cache_key in self._lut_cache:
            return self._lut_cache[cache_key]
        
        # 生成LUT
        start_time = time.time()
        lut = self._generate_lut_from_params(params, size)
        
        # 缓存LUT
        self._cache_lut(cache_key, lut)
        
        print(f"LUT生成耗时: {time.time() - start_time:.3f}秒")
        return lut
    
    def _generate_lut_from_params(self, params: ColorGradingParams, size: int) -> LUT3D:
        """根据参数生成LUT"""
        lut_data = np.zeros((size**3, 3), dtype=np.float32)
        
        # 生成输入颜色立方体
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    idx = i * size**2 + j * size + k
                    
                    # 输入颜色值 [0, 1]
                    input_color = np.array([
                        i / (size - 1),
                        j / (size - 1), 
                        k / (size - 1)
                    ], dtype=np.float32)
                    
                    # 创建虚拟图像数据
                    virtual_image = ImageData(
                        array=input_color.reshape(1, 1, 3),
                        width=1,
                        height=1,
                        channels=3,
                        dtype=np.float32,
                        color_space="ACEScg",
                        file_path="",
                        is_proxy=True,
                        proxy_scale=1.0
                    )
                    
                    # 应用调色管道
                    result = self.the_enlarger.apply_full_pipeline(virtual_image, params)
                    
                    # 存储输出颜色
                    lut_data[idx] = result.array[0, 0, :3]
        
        return LUT3D(size=size, data=lut_data)
    
    def apply_lut_to_image(self, image: ImageData, lut: LUT3D) -> ImageData:
        """将LUT应用到图像"""
        if image.array is None:
            return image
        
        # 应用LUT
        result_array = lut.apply_to_image(image.array)
        
        # 创建结果图像
        result_image = ImageData(
            array=result_array,
            width=image.width,
            height=image.height,
            channels=image.channels,
            dtype=image.dtype,
            color_space=image.color_space,
            icc_profile=image.icc_profile,
            metadata=image.metadata,
            file_path=image.file_path,
            is_proxy=image.is_proxy,
            proxy_scale=image.proxy_scale
        )
        
        return result_image
    
    def save_lut(self, lut: LUT3D, format: str, path: str) -> bool:
        """保存LUT文件"""
        try:
            if format.lower() == "cube":
                return self._save_cube_lut(lut, path)
            elif format.lower() == "3dl":
                return self._save_3dl_lut(lut, path)
            else:
                print(f"不支持的LUT格式: {format}")
                return False
        except Exception as e:
            print(f"保存LUT失败: {e}")
            return False
    
    def load_lut(self, path: str) -> Optional[LUT3D]:
        """加载LUT文件"""
        try:
            path_lower = path.lower()
            if path_lower.endswith(".cube"):
                return self._load_cube_lut(path)
            elif path_lower.endswith(".3dl"):
                return self._load_3dl_lut(path)
            else:
                print(f"不支持的LUT格式: {path}")
                return None
        except Exception as e:
            print(f"加载LUT失败: {e}")
            return None
    
    def _save_cube_lut(self, lut: LUT3D, path: str) -> bool:
        """保存CUBE格式LUT"""
        try:
            with open(path, 'w') as f:
                f.write("# DiVERE Generated LUT\n")
                f.write(f"LUT_3D_SIZE {lut.size}\n")
                f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
                f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
                f.write("\n")
                
                # 写入LUT数据
                for i in range(lut.size**3):
                    r, g, b = lut.data[i]
                    f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
            
            return True
        except Exception as e:
            print(f"保存CUBE LUT失败: {e}")
            return False
    
    def _load_cube_lut(self, path: str) -> Optional[LUT3D]:
        """加载CUBE格式LUT"""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            size = 32  # 默认大小
            data = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    continue
                elif line.startswith('LUT_3D_SIZE'):
                    size = int(line.split()[1])
                elif line.startswith('DOMAIN_MIN') or line.startswith('DOMAIN_MAX'):
                    continue
                elif line and not line.startswith('#'):
                    # 解析RGB值
                    values = line.split()
                    if len(values) >= 3:
                        r = float(values[0])
                        g = float(values[1])
                        b = float(values[2])
                        data.append([r, g, b])
            
            if len(data) == size**3:
                return LUT3D(size=size, data=np.array(data, dtype=np.float32))
            else:
                print(f"LUT数据不匹配: 期望{size**3}个值，实际{len(data)}个")
                return None
                
        except Exception as e:
            print(f"加载CUBE LUT失败: {e}")
            return None
    
    def _save_3dl_lut(self, lut: LUT3D, path: str) -> bool:
        """保存3DL格式LUT"""
        try:
            with open(path, 'w') as f:
                f.write(f"{lut.size} {lut.size} {lut.size}\n")
                
                # 写入LUT数据
                for i in range(lut.size**3):
                    r, g, b = lut.data[i]
                    f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")
            
            return True
        except Exception as e:
            print(f"保存3DL LUT失败: {e}")
            return False
    
    def _load_3dl_lut(self, path: str) -> Optional[LUT3D]:
        """加载3DL格式LUT"""
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            # 第一行包含尺寸信息
            size_line = lines[0].strip().split()
            if len(size_line) >= 3:
                size = int(size_line[0])
            else:
                size = 32
            
            data = []
            
            for line in lines[1:]:
                line = line.strip()
                if line:
                    values = line.split()
                    if len(values) >= 3:
                        r = float(values[0])
                        g = float(values[1])
                        b = float(values[2])
                        data.append([r, g, b])
            
            if len(data) == size**3:
                return LUT3D(size=size, data=np.array(data, dtype=np.float32))
            else:
                print(f"LUT数据不匹配: 期望{size**3}个值，实际{len(data)}个")
                return None
                
        except Exception as e:
            print(f"加载3DL LUT失败: {e}")
            return None
    
    def _get_params_hash(self, params: ColorGradingParams) -> str:
        """获取参数的哈希值用于缓存"""
        # 简化的哈希计算
        param_str = f"{params.density_gamma}_{params.density_gain}_{params.density_dmax}_{params.rgb_gains}_{params.correction_matrix_file}_{params.debug_mode}_{params.enable_density_inversion}_{params.enable_correction_matrix}_{params.enable_rgb_gains}_{params.enable_density_curve}"
        return str(hash(param_str))
    
    def _cache_lut(self, key: str, lut: LUT3D):
        """缓存LUT"""
        # 简单的LRU缓存
        if len(self._lut_cache) >= self._max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._lut_cache))
            del self._lut_cache[oldest_key]
        
        self._lut_cache[key] = lut
    
    def clear_cache(self):
        """清空LUT缓存"""
        self._lut_cache.clear()
    
    def get_cache_info(self) -> dict:
        """获取缓存信息"""
        return {
            "cache_size": len(self._lut_cache),
            "max_cache_size": self._max_cache_size,
            "cached_keys": list(self._lut_cache.keys())
        } 