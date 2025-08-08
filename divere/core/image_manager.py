"""
图像管理模块
负责图像的导入、代理生成、缓存管理
"""

import os
import hashlib
from typing import Optional, Dict, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import imageio

from .data_types import ImageData


class ImageManager:
    """图像管理器"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._proxy_cache: Dict[str, ImageData] = {}
        self._max_cache_size = 10  # 最大缓存图像数量
        
    def load_image(self, file_path: str) -> ImageData:
        """加载图像文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {file_path}")
        ext = file_path.suffix.lower()
        
        # 优先使用PIL处理TIFF以正确识别通道顺序（如RGBA/ARGB/CMYK等）
        if ext in [".tif", ".tiff"]:
            try:
                pil_image = Image.open(file_path)
                mode = pil_image.mode  # 例如: 'RGB', 'RGBA', 'CMYK', 'I;16', 'F', 'LA' 等
                bands = pil_image.getbands()  # 例如: ('R','G','B','A') 或 ('A','R','G','B') 或 ('C','M','Y','K')

                # 若为CMYK或其它非RGB空间，先转换到RGB
                if mode in ["CMYK", "YCbCr", "LAB"]:
                    pil_image = pil_image.convert("RGB")
                    mode = pil_image.mode
                    bands = pil_image.getbands()

                image = np.array(pil_image)

                # 归一化到[0,1]
                if image.dtype != np.float32:
                    image = image.astype(np.float32)
                    if image.max() > 1.0:
                        # 根据位深度推断范围
                        if image.max() > 255:
                            image /= 65535.0
                        else:
                            image /= 255.0

                # 灰度转为单通道形状 (H,W,1)
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]

                # 处理4通道的通道顺序：优先识别Alpha；否则启发式识别红外IR通道（更“平滑/平均”）并移到最后
                if image.ndim == 3 and image.shape[2] == 4:
                    print(f"[ImageManager] 检测到4通道TIFF: {file_path.name}, mode={mode}, bands={bands}")
                    handled = False
                    # 1) 若bands可用且包含A，按RGBA重排
                    if bands is not None:
                        try:
                            band_list = list(bands)
                            if 'A' in band_list:
                                alpha_idx = band_list.index('A')
                                if set(['R','G','B']).issubset(set(band_list)):
                                    r_idx = band_list.index('R')
                                    g_idx = band_list.index('G')
                                    b_idx = band_list.index('B')
                                    image = image[..., [r_idx, g_idx, b_idx, alpha_idx]]
                                    print(f"[ImageManager] 通过bands识别Alpha通道(index={alpha_idx})，已重排为RGBA顺序")
                                else:
                                    # 仅将Alpha放末尾
                                    rgb_indices = [i for i in range(4) if i != alpha_idx]
                                    image = image[..., rgb_indices + [alpha_idx]]
                                    print(f"[ImageManager] 通过bands识别Alpha通道(index={alpha_idx})，已将Alpha移至最后")
                                handled = True
                        except Exception:
                            handled = False
                    
                    if not handled:
                        # 2) 启发式识别IR通道：其方差/Laplacian方差明显更低（更平滑/平均）
                        # 采样以提速
                        sample = image[::8, ::8, :]
                        H, W, _ = sample.shape
                        ch = sample.reshape(-1, 4)
                        var_spatial = ch.var(axis=0)
                        # 简单梯度能量（近似边缘能量）
                        gx = np.diff(sample, axis=1, prepend=sample[:, :1, :])
                        gy = np.diff(sample, axis=0, prepend=sample[:1, :, :])
                        edge_energy = (gx**2 + gy**2).mean(axis=(0, 1))
                        score = var_spatial + 0.5 * edge_energy
                        candidate = int(np.argmin(score))
                        # 与次小值比较，确保明显更低
                        sorted_scores = np.sort(score)
                        if sorted_scores[0] < 0.5 * sorted_scores[1]:
                            # 将IR放到最后，其余通道保持原相对顺序
                            order = [i for i in range(4) if i != candidate] + [candidate]
                            image = image[..., order]
                            print(f"[ImageManager] 通过启发式识别IR通道(index={candidate})，score={score.round(6).tolist()}，已重排为RGB+IR")
                        else:
                            print(f"[ImageManager] 启发式无法明确识别IR通道，保持原通道顺序，score={score.round(6).tolist()}")

                    # 3) 若存在Alpha通道，则在导入时直接丢弃Alpha，避免影响后续流程
                    drop_alpha = False
                    alpha_index = None
                    # 明确的bands包含A
                    if bands is not None and 'A' in list(bands):
                        # 若前面按bands重排过，此时A应在末位；稳妥起见再次定位索引
                        alpha_index = list(bands).index('A')
                        # 若已重排至RGBA，alpha_index应为3；否则按当前位置删除
                        drop_alpha = True
                    else:
                        # 启发式判断Alpha（近乎常量且接近0或1）
                        sample = image[::8, ::8, :]
                        ch = sample.reshape(-1, 4)
                        vars_ = ch.var(axis=0)
                        means_ = ch.mean(axis=0)
                        for idx in range(4):
                            if vars_[idx] < 1e-6 and (means_[idx] < 0.01 or means_[idx] > 0.99):
                                alpha_index = idx
                                drop_alpha = True
                                break
                    if drop_alpha and alpha_index is not None and 0 <= alpha_index < 4:
                        image = np.delete(image, alpha_index, axis=2)
                        print(f"[ImageManager] 检测到Alpha通道(index={alpha_index})，已在导入时移除。当前shape={image.shape}")

                # 创建ImageData并返回
                image_data = ImageData(
                    array=image,
                    color_space="Film_KodakRGB_Linear",
                    file_path=str(file_path),
                    is_proxy=False,
                    proxy_scale=1.0
                )
                return image_data
            except Exception as e:
                # 回退到OpenCV路径
                pass
        
        # 尝试使用OpenCV加载（非TIFF优先走此分支）
        try:
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV无法加载图像")
            
            # OpenCV使用BGR(A)格式，转换为RGB(A)
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    print(f"[ImageManager] 检测到4通道图像(BGRA→RGBA): {file_path.name}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    # 对OpenCV读取的4通道图也进行IR启发式识别（若不存在明显Alpha）
                    sample = image[::8, ::8, :].astype(np.float32) / (255.0 if image.dtype!=np.float32 and image.max()>1.0 else 1.0)
                    ch = sample.reshape(-1, 4)
                    var_spatial = ch.var(axis=0)
                    gx = np.diff(sample, axis=1, prepend=sample[:, :1, :])
                    gy = np.diff(sample, axis=0, prepend=sample[:1, :, :])
                    edge_energy = (gx**2 + gy**2).mean(axis=(0, 1))
                    score = var_spatial + 0.5 * edge_energy
                    candidate = int(np.argmin(score))
                    sorted_scores = np.sort(score)
                    # 若第4通道近似透明掩码（极低方差且均值接近0或1），放行不改。否则按IR处理。
                    means = ch.mean(axis=0)
                    is_alpha_like = (var_spatial[candidate] < 1e-6) and (means[candidate] < 0.01 or means[candidate] > 0.99)
                    if (sorted_scores[0] < 0.5 * sorted_scores[1]) and not is_alpha_like:
                        order = [i for i in range(4) if i != candidate] + [candidate]
                        image = image[..., order]
                        print(f"[ImageManager] 通过启发式识别IR通道(index={candidate})，score={score.round(6).tolist()}，已重排为RGB+IR")
                    else:
                        print(f"[ImageManager] 4通道保持顺序，score={score.round(6).tolist()}，alpha_like={bool(is_alpha_like)}")

                    # 导入时移除Alpha（若存在）
                    # 再次采用启发式：近乎常量且接近0/1的通道视为Alpha
                    sample2 = image[::8, ::8, :].astype(np.float32) / (255.0 if image.dtype!=np.float32 and image.max()>1.0 else 1.0)
                    ch2 = sample2.reshape(-1, 4)
                    vars2 = ch2.var(axis=0)
                    means2 = ch2.mean(axis=0)
                    alpha_idx = None
                    for idx in range(4):
                        if vars2[idx] < 1e-6 and (means2[idx] < 0.01 or means2[idx] > 0.99):
                            alpha_idx = idx
                            break
                    if alpha_idx is not None:
                        image = np.delete(image, alpha_idx, axis=2)
                        print(f"[ImageManager] 检测到Alpha通道(index={alpha_idx})，已在导入时移除。当前shape={image.shape}")
            
            # 转换为float32并归一化到[0,1]
            if image.dtype != np.float32:
                original_dtype = image.dtype
                image = image.astype(np.float32)
                
                # 根据原始数据类型进行正确的归一化
                if original_dtype == np.uint8:
                    # 8bit图像: 0-255 -> 0-1
                    image /= 255.0
                elif original_dtype == np.uint16:
                    # 16bit图像: 0-65535 -> 0-1
                    image /= 65535.0
                elif original_dtype == np.uint32:
                    # 32bit图像: 0-4294967295 -> 0-1
                    image /= 4294967295.0
                elif original_dtype == np.int16:
                    # 16bit有符号: -32768-32767 -> 0-1
                    image = (image + 32768) / 65535.0
                elif original_dtype == np.int32:
                    # 32bit有符号: -2147483648-2147483647 -> 0-1
                    image = (image + 2147483648) / 4294967295.0
                elif image.max() > 1.0:
                    # 其他情况，使用最大值归一化
                    image /= image.max()
            
        except Exception as e:
            # 如果OpenCV失败，尝试使用PIL
            try:
                pil_image = Image.open(file_path)
                original_mode = pil_image.mode
                image = np.array(pil_image, dtype=np.float32)
                
                # 根据PIL模式进行正确的归一化
                if original_mode in ['L', 'RGB', 'RGBA']:
                    # 8bit图像
                    if image.max() > 1.0:
                        image /= 255.0
                elif original_mode in ['I', 'I;16']:
                    # 16bit图像
                    if image.max() > 1.0:
                        image /= 65535.0
                elif original_mode in ['F']:
                    # 32bit浮点图像，通常已经是0-1范围
                    pass
                elif image.max() > 1.0:
                    # 其他情况，使用最大值归一化
                    image /= image.max()
                
                # 处理灰度图像
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]
                    
            except Exception as e2:
                raise RuntimeError(f"无法加载图像文件: {e}, {e2}")
        
        # 如果存在Alpha通道，暂不丢弃，但在后续色彩空间转换时会自动忽略
        # 创建ImageData对象（默认Film_KodakRGB_Linear，色彩空间将由用户手动选择）
        image_data = ImageData(
            array=image,
            color_space="Film_KodakRGB_Linear",  # 默认Film_KodakRGB_Linear
            file_path=str(file_path),
            is_proxy=False,
            proxy_scale=1.0
        )
        
        return image_data
    
    def generate_proxy(self, image: ImageData, max_size: Tuple[int, int] = (1920, 1080)) -> ImageData:
        """生成代理图像"""
        # 移除这个检查，允许对代理图像进行进一步缩放
        # if image.is_proxy:
        #     return image
        
        # 计算缩放比例
        h, w = image.height, image.width
        max_w, max_h = max_size
        
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)  # 不放大图像
        
        if scale >= 1.0:
            # 图像已经足够小，直接返回
            return image
        
        # 计算新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 使用OpenCV进行高质量缩放
        proxy_array = cv2.resize(
            image.array, 
            (new_w, new_h), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        # 创建代理ImageData
        proxy_data = ImageData(
            array=proxy_array,
            color_space=image.color_space,
            icc_profile=image.icc_profile,
            metadata=image.metadata,
            file_path=image.file_path,
            is_proxy=True,
            proxy_scale=scale
        )
        
        return proxy_data
    
    def get_cached_proxy(self, image_id: str) -> Optional[ImageData]:
        """获取缓存的代理图像"""
        return self._proxy_cache.get(image_id)
    
    def cache_proxy(self, image_id: str, proxy: ImageData):
        """缓存代理图像"""
        # 简单的LRU缓存实现
        if len(self._proxy_cache) >= self._max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._proxy_cache))
            del self._proxy_cache[oldest_key]
        
        self._proxy_cache[image_id] = proxy
    
    def clear_cache(self):
        """清空缓存"""
        self._proxy_cache.clear()
    
    def get_image_id(self, file_path: str) -> str:
        """生成图像唯一标识符"""
        # 使用文件路径和修改时间的哈希作为ID
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_image(self, image_data: ImageData, output_path: str, quality: int = 95, bit_depth: int = 8):
        """保存图像"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保图像数据在[0,1]范围内
        image_array = np.clip(image_data.array, 0, 1)
        
        # 根据位深度和文件格式选择保存类型
        ext = output_path.suffix.lower()
        
        if bit_depth == 16 and ext in ['.png', '.tiff', '.tif']:
            # 16bit保存
            image_array = (image_array * 65535).astype(np.uint16)
        else:
            # 8bit保存
            image_array = (image_array * 255).astype(np.uint8)
        
        # 根据文件扩展名选择保存格式
        ext = output_path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg']:
            # JPEG格式
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # 转换为BGR用于OpenCV
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(output_path), image_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        elif ext in ['.png']:
            # PNG格式
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image)
            else:
                cv2.imwrite(str(output_path), image_array)
        
        elif ext in ['.tiff', '.tif']:
            # TIFF格式
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), bgr_image)
            else:
                cv2.imwrite(str(output_path), image_array)
        
        else:
            # 默认使用PIL保存
            pil_image = Image.fromarray(image_array)
            pil_image.save(output_path, quality=quality)
    
    def get_supported_formats(self) -> list:
        """获取支持的图像格式"""
        return [
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', 
            '.bmp', '.webp', '.exr', '.hdr'
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_formats() 