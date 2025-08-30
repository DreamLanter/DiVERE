# DiVERE Processing Pipeline Documentation

This document contains the core processing code for both preview and export pipelines in the DiVERE film color correction tool.

## Preview Pipeline

### Entry Point
**File**: `divere/core/the_enlarger.py`
**Function**: `process_image_for_preview()`

```python
def process_image_for_preview(self, image: ImageData, params: ColorGradingParams,
                            input_colorspace_transform: Optional[np.ndarray] = None,
                            output_colorspace_transform: Optional[np.ndarray] = None) -> ImageData:
    """预览图像处理"""
    use_optimization = not for_export  # True for preview
    
    return self.pipeline_processor.apply_preview_pipeline(
        image, params, input_colorspace_transform, output_colorspace_transform,
        use_optimization=use_optimization,
        include_curve=True
    )
```

### Core Processing Steps

#### 1. Proxy Image Creation
**File**: `divere/core/pipeline_processor.py`
**Function**: `_create_preview_proxy()`

```python
def _create_preview_proxy(self, image_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """创建预览代理图像（优化版）"""
    h, w = image_array.shape[:2]
    max_dim = max(h, w)
    
    scale_factor = min(1.0, self.preview_config.preview_max_size / max_dim)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # 大幅度缩放使用两步法
    scale_ratio = scale_factor
    if scale_ratio < 0.3:
        intermediate_factor = 0.5
        intermediate_h = int(h * intermediate_factor)
        intermediate_w = int(w * intermediate_factor)
        
        temp_proxy = cv2.resize(image_array, (intermediate_w, intermediate_h), 
                              interpolation=cv2.INTER_CUBIC)
        proxy = cv2.resize(temp_proxy, (new_w, new_h), 
                         interpolation=cv2.INTER_CUBIC)
    else:
        proxy = cv2.resize(image_array, (new_w, new_h), 
                         interpolation=cv2.INTER_CUBIC)
    
    return proxy, scale_factor
```

#### 2. Density Inversion (if enabled)
**File**: `divere/core/math_ops.py`
**Function**: `density_inversion()`

```python
def density_inversion(self, image_array: np.ndarray, 
                     gamma: float, dmax: float, 
                     pivot: float = 0.9, use_optimization: bool = True,
                     use_gpu: bool = True, use_parallel: bool = True) -> np.ndarray:
    
    # GPU acceleration for preview (use_optimization=True)
    if (use_gpu and use_optimization and
        self.gpu_accelerator and 
        self.preview_config.should_use_gpu(image_array.size)):
        return self.gpu_accelerator.density_inversion_gpu(image_array, gamma, dmax, pivot)
    
    # CPU processing
    if use_optimization:
        return self._density_inversion_optimized_lut(image_array, gamma, dmax, pivot, use_parallel)
    else:
        return self._density_inversion_direct(image_array, gamma, dmax, pivot, use_parallel)
```

**Core LUT-based inversion logic:**
```python
def _density_inversion_optimized_lut(self, image_array: np.ndarray, gamma: float, 
                                   dmax: float, pivot: float, use_parallel: bool) -> np.ndarray:
    
    # Generate LUT for optimization
    lut_size = 65536
    input_values = np.linspace(0.0, 1.0, lut_size)
    
    # Apply density inversion formula
    safe = np.maximum(input_values, 1e-10)
    original_density = -np.log10(safe)
    adjusted_density = pivot + (original_density - pivot) * gamma - dmax
    lut = np.clip(np.power(10.0, adjusted_density), 0.0, 1.0)
    
    # Apply LUT to image
    img_clipped = np.clip(image_array, 0.0, 1.0)
    indices = np.round(img_clipped * (lut_size - 1)).astype(np.uint16)
    
    return np.take(lut, indices)
```

#### 3. Preview LUT Pipeline
**File**: `divere/core/pipeline_processor.py`
**Function**: `_apply_preview_lut_pipeline()`

```python
def _apply_preview_lut_pipeline(self, proxy_array: np.ndarray, params: ColorGradingParams,
                               include_curve: bool, profile: Dict[str, float]) -> np.ndarray:
    
    # 转为密度空间
    density_array = self.math_ops.linear_to_density(proxy_array)
    
    # 密度校正矩阵
    if params.enable_density_matrix:
        matrix = self._get_density_matrix_from_params(params)
        if matrix is not None and not np.allclose(matrix, np.eye(3)):
            density_array = self.math_ops.apply_density_matrix(
                density_array, matrix, params.density_dmax, use_parallel=False
            )
    
    # RGB曝光调整
    if params.enable_rgb_gains:
        density_array = self.math_ops.apply_rgb_gains(
            density_array, params.rgb_gains, use_parallel=False
        )
    
    # 密度曲线调整
    if include_curve and params.enable_density_curve:
        curve_points = params.curve_points if not self._is_default_curve(params.curve_points) else None
        
        channel_curves = {}
        if not self._is_default_curve(params.curve_points_r):
            channel_curves['r'] = params.curve_points_r
        if not self._is_default_curve(params.curve_points_g):
            channel_curves['g'] = params.curve_points_g
        if not self._is_default_curve(params.curve_points_b):
            channel_curves['b'] = params.curve_points_b
        
        if curve_points or channel_curves:
            density_array = self.math_ops.apply_density_curve(
                density_array, curve_points, channel_curves, use_parallel=False,
                use_optimization=True  # Preview uses LUT optimization
            )
    
    # 转回线性
    result_array = self.math_ops.density_to_linear(density_array)
    
    return result_array
```

#### 4. Linear ↔ Density Space Conversion
**File**: `divere/core/math_ops.py`

```python
def linear_to_density(self, linear_array: np.ndarray) -> np.ndarray:
    """线性转密度：density = -log10(linear)"""
    safe = np.maximum(linear_array, 1e-10)
    return -np.log10(safe)

def density_to_linear(self, density_array: np.ndarray) -> np.ndarray:
    """密度转线性：linear = 10^(-density) = exp(-density * ln(10))"""
    ln10 = np.log(10.0)
    result = np.exp(-density_array * ln10)
    return np.clip(result, 0.0, 1.0)
```

#### 5. Density Matrix Application
**File**: `divere/core/math_ops.py`

```python
def apply_density_matrix(self, density_array: np.ndarray, matrix: np.ndarray,
                        dmax: float, pivot: float = 0.9, use_parallel: bool = True) -> np.ndarray:
    
    # 准备输入：添加dmax偏移
    input_density = density_array + dmax
    
    # 多通道图像，正常处理
    reshaped = input_density.reshape(-1, input_density.shape[-1])
    
    if input_density.shape[-1] == 3:
        # RGB图像，直接应用变换
        adjusted = pivot + np.dot(reshaped - pivot, matrix.T)
        result = adjusted.reshape(density_array.shape) - dmax
    else:
        # 其他通道数，仅处理前3个通道
        rgb_part = reshaped[:, :3]
        adjusted_rgb = pivot + np.dot(rgb_part - pivot, matrix.T)
        adjusted = reshaped.copy()
        adjusted[:, :3] = adjusted_rgb
        result = adjusted.reshape(density_array.shape) - dmax
    
    return result
```

#### 6. RGB Gains Application
**File**: `divere/core/math_ops.py`

```python
def apply_rgb_gains(self, density_array: np.ndarray, rgb_gains: Tuple[float, float, float],
                   use_parallel: bool = True) -> np.ndarray:
    """应用RGB增益调整（在密度空间）"""
    if not rgb_gains or all(g == 0.0 for g in rgb_gains):
        return density_array
        
    result = density_array.copy()
    
    # RGB增益在密度空间的应用：正增益降低密度（变亮），负增益增加密度（变暗）
    num_channels = min(result.shape[2], len(rgb_gains))
    for i in range(num_channels):
        result[:, :, i] -= rgb_gains[i]
        
    return result
```

#### 7. Density Curve Application (Preview - LUT Mode)
**File**: `divere/core/math_ops.py`

```python
def _apply_curves_merged_lut(self, density_array: np.ndarray,
                           curve_points: Optional[List[Tuple[float, float]]],
                           channel_curves: Optional[Dict[str, List[Tuple[float, float]]]],
                           lut_size: int) -> np.ndarray:
    """合并曲线LUT实现 - 内存访问优化版本"""
    
    # 预计算常量（只算一次）
    inv_range = 1.0 / self._LOG65536
    log65536 = self._LOG65536
    lut_scale = lut_size - 1
    
    # 原地操作，避免拷贝
    result = density_array
    
    # 预计算所有通道的LUT（批量操作）
    channel_luts = []
    channel_map = ['r', 'g', 'b']
    
    for channel_name in channel_map:
        merged_lut = self._get_merged_channel_lut(
            curve_points, 
            channel_curves.get(channel_name) if channel_curves else None,
            lut_size
        )
        channel_luts.append(merged_lut)
    
    # 一次性归一化所有通道（向量化）
    normalized = 1.0 - np.clip(result * inv_range, 0.0, 1.0)
    indices = (normalized * lut_scale).astype(np.uint16, copy=False)
    
    # 向量化处理所有通道
    for channel_idx, merged_lut in enumerate(channel_luts):
        if merged_lut is not None:
            channel_indices = indices[:, :, channel_idx]
            curve_output = np.take(merged_lut, channel_indices)
            result[:, :, channel_idx] = (1.0 - curve_output) * log65536
    
    return result
```

**LUT Generation:**
```python
def _generate_curve_lut_fast(self, curve_points: List[Tuple[float, float]], lut_size: int) -> np.ndarray:
    """快速生成曲线LUT"""
    points_array = np.array(curve_points, dtype=np.float64)
    
    x_values = np.linspace(0.0, 1.0, lut_size, dtype=np.float64)
    y_values = np.interp(x_values, points_array[:, 0], points_array[:, 1])
    
    return y_values.astype(np.float64)
```

---

## Export Pipeline

### Entry Point
**File**: `divere/ui/main_window.py`
**Function**: Export functions

```python
def _export_image_to_file(self, file_path: str, params: ColorGradingParams, 
                         input_colorspace_transform: Optional[np.ndarray],
                         output_colorspace_transform: Optional[np.ndarray]) -> bool:
    
    working_image = self.app_context.current_image.copy()
    
    # 导出模式：提升为float64精度，确保全程高精度计算
    if working_image.array is not None:
        working_image.array = working_image.array.astype(np.float64)
        working_image.dtype = np.float64
    
    # 密度反相处理（如果需要）
    if params.enable_density_inversion:
        working_image.array = self.app_context.enlarger.math_ops.density_inversion(
            working_image.array, idt_gamma, use_optimization=False  # Export uses high precision
        )
    
    # 应用完整精度处理管线
    processed_image = self.app_context.enlarger.pipeline_processor.apply_full_precision_pipeline(
        working_image, params, input_colorspace_transform, output_colorspace_transform,
        use_optimization=False, include_curve=True  # Export uses high precision
    )
    
    return processed_image
```

### Core Processing Steps

#### 1. Full Precision Pipeline Entry
**File**: `divere/core/pipeline_processor.py`
**Function**: `apply_full_precision_pipeline()`

```python
def apply_full_precision_pipeline(self, image: ImageData, params: ColorGradingParams,
                                 input_colorspace_transform: Optional[np.ndarray] = None,
                                 output_colorspace_transform: Optional[np.ndarray] = None,
                                 include_curve: bool = True,
                                 use_optimization: bool = True,  # Export passes False
                                 chunked: Optional[bool] = None,
                                 tile_size: Optional[Tuple[int, int]] = None,
                                 max_workers: Optional[int] = None) -> ImageData:
    
    src_array = image.array
    h, w, c = src_array.shape
    
    # 自动决定是否分块处理
    should_chunk = (h * w) > self.full_pipeline_chunk_threshold  # 16MP threshold
    
    if should_chunk and chunked is not False:
        return self._apply_full_precision_chunked(
            image, params, input_colorspace_transform, output_colorspace_transform,
            include_curve, use_optimization, tile_size, max_workers
        )
    else:
        return self._apply_full_precision_single_pass(
            image, params, input_colorspace_transform, output_colorspace_transform,
            include_curve, use_optimization
        )
```

#### 2. Chunked Processing (for large images)
**File**: `divere/core/pipeline_processor.py`

```python
def _apply_full_precision_chunked(self, image: ImageData, params: ColorGradingParams,
                                input_colorspace_transform: Optional[np.ndarray],
                                output_colorspace_transform: Optional[np.ndarray],
                                include_curve: bool, use_optimization: bool,
                                tile_size: Optional[Tuple[int, int]],
                                max_workers: Optional[int]) -> ImageData:
    
    src_array = image.array
    h, w, c = src_array.shape
    
    # 分块处理参数
    th, tw = tile_size or self.full_pipeline_tile_size  # Default: (2048, 2048)
    
    # 计算分块数量
    tiles_h = (h + th - 1) // th
    tiles_w = (w + tw - 1) // tw
    
    # 创建输出数组
    result_array = np.zeros_like(src_array)
    
    def process_tile(tile_coords: Tuple[int, int, int, int]) -> Tuple[Tuple[int,int,int,int], np.ndarray]:
        sh, eh, sw, ew = tile_coords
        block = src_array[sh:eh, sw:ew, :].copy()
        
        # 应用数学管线到块
        processed_block = self.math_ops.apply_full_math_pipeline(
            block, params, include_curve,
            enable_density_inversion=False, use_optimization=use_optimization
        )
        
        return (tile_coords, processed_block)
    
    # 并行处理所有块
    tile_coords_list = [
        (i * th, min((i + 1) * th, h), j * tw, min((j + 1) * tw, w))
        for i in range(tiles_h) for j in range(tiles_w)
    ]
    
    executor = ThreadPoolExecutor(max_workers=max_workers or self.full_pipeline_max_workers)
    results = list(executor.map(process_tile, tile_coords_list))
    
    # 重组结果
    for tile_coords, processed_block in results:
        sh, eh, sw, ew = tile_coords
        result_array[sh:eh, sw:ew, :] = processed_block
    
    return image.copy_with_new_array(result_array)
```

#### 3. High Precision Density Inversion (Export Mode)
**File**: `divere/core/math_ops.py`

```python
def _density_inversion_direct(self, image_array: np.ndarray, gamma: float, 
                            dmax: float, pivot: float, use_parallel: bool) -> np.ndarray:
    """直接版本的密度反相（高精度）"""
    
    # 防止log(0)的安全处理
    safe = np.maximum(image_array, 1e-10)
    
    # 计算原始密度
    original_density = -np.log10(safe)
    
    # 应用gamma和dmax调整
    adjusted_density = pivot + (original_density - pivot) * gamma - dmax
    
    # 转换回线性空间
    result = np.power(10.0, adjusted_density)
    
    # 裁剪到有效范围
    return np.clip(result, 0.0, 1.0)
```

#### 4. Full Math Pipeline
**File**: `divere/core/math_ops.py`

```python
def apply_full_math_pipeline(self, image_array: np.ndarray, params: ColorGradingParams,
                           include_curve: bool = True,
                           enable_density_inversion: bool = True,
                           use_optimization: bool = True) -> np.ndarray:
    """完整数学管线"""
    
    result_array = image_array.copy()
    
    # 1. 密度反相（如果启用）
    if enable_density_inversion and params.enable_density_inversion:
        result_array = self.density_inversion(
            result_array, params.density_gamma, params.density_dmax,
            params.density_pivot, use_optimization=use_optimization
        )
    
    # 2. 转换到密度空间
    density_array = self.linear_to_density(result_array)
    
    # 3. 应用密度校正矩阵
    if params.enable_density_matrix:
        matrix = self._get_density_matrix(params)
        if matrix is not None and not np.allclose(matrix, np.eye(3)):
            density_array = self.apply_density_matrix(
                density_array, matrix, params.density_dmax, params.density_pivot,
                use_parallel=True
            )
    
    # 4. 应用RGB增益
    if params.enable_rgb_gains:
        density_array = self.apply_rgb_gains(
            density_array, params.rgb_gains, use_parallel=True
        )
    
    # 5. 应用密度曲线
    if include_curve and params.enable_density_curve:
        curve_points = params.curve_points if not self._is_default_curve(params.curve_points) else None
        
        channel_curves = {}
        if not self._is_default_curve(params.curve_points_r):
            channel_curves['r'] = params.curve_points_r
        if not self._is_default_curve(params.curve_points_g):
            channel_curves['g'] = params.curve_points_g
        if not self._is_default_curve(params.curve_points_b):
            channel_curves['b'] = params.curve_points_b
        
        if curve_points or channel_curves:
            density_array = self.apply_density_curve(
                density_array, curve_points, channel_curves,
                use_parallel=True, use_optimization=use_optimization
            )
    
    # 6. 转换回线性空间
    result_array = self.density_to_linear(density_array)
    
    return result_array
```

#### 5. High Precision Density Curve Application (Export Mode)
**File**: `divere/core/math_ops.py`

```python
def _apply_curves_pure_interpolation(self, density_array: np.ndarray,
                                   curve_points: Optional[List[Tuple[float, float]]],
                                   channel_curves: Optional[Dict[str, List[Tuple[float, float]]]]) -> np.ndarray:
    """纯插值模式的密度曲线应用（无LUT，用于导出）"""
    
    result = density_array.copy()
    inv_range = 1.0 / self._LOG65536
    log65536 = self._LOG65536
    
    # 处理每个通道
    for channel_idx in range(min(3, result.shape[2])):
        channel_name = ['r', 'g', 'b'][channel_idx]
        channel_data = result[:, :, channel_idx]
        
        # 收集该通道的所有曲线
        curves_to_apply = []
        
        # RGB通用曲线
        if curve_points and len(curve_points) >= 2:
            curves_to_apply.append(curve_points)
        
        # 单通道曲线
        if (channel_curves and channel_name in channel_curves and 
            channel_curves[channel_name] and len(channel_curves[channel_name]) >= 2):
            curves_to_apply.append(channel_curves[channel_name])
        
        # 逐个应用曲线
        for curve in curves_to_apply:
            # 归一化到[0,1]
            normalized = 1.0 - np.clip(channel_data * inv_range, 0.0, 1.0)
            
            # 纯数学插值（无LUT量化）
            x_points = np.array([p[0] for p in curve], dtype=np.float64)
            y_points = np.array([p[1] for p in curve], dtype=np.float64)
            interpolated = np.interp(normalized, x_points, y_points)
            
            # 映射回密度空间
            channel_data = (1.0 - interpolated) * log65536
        
        result[:, :, channel_idx] = channel_data
    
    return result
```

#### 6. Color Space Transforms
**File**: `divere/core/pipeline_processor.py`

```python
def _apply_colorspace_transform(self, image_array: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """应用颜色空间变换矩阵"""
    original_shape = image_array.shape
    
    # 重塑为 [N, 3] 进行矩阵乘法
    reshaped = image_array.reshape(-1, original_shape[-1])
    
    if original_shape[-1] == 3:
        # RGB图像，直接变换
        transformed = np.dot(reshaped, transform_matrix.T)
    else:
        # 其他通道数，只变换前3个通道
        rgb_part = reshaped[:, :3]
        transformed_rgb = np.dot(rgb_part, transform_matrix.T)
        transformed = reshaped.copy()
        transformed[:, :3] = transformed_rgb
    
    result = transformed.reshape(original_shape)
    
    # 裁剪到有效范围
    return np.clip(result, 0.0, 1.0)
```

---

## Key Algorithmic Differences

### Preview Pipeline Optimizations
- **Early downsampling**: Reduces computation by 90%+ for large images
- **LUT-based processing**: 32K lookup tables for density curves
- **GPU acceleration**: Metal/OpenCL/CUDA when available
- **Single-threaded**: Optimized for small proxy images
- **Float32 precision**: Acceptable for real-time preview

### Export Pipeline Precision
- **Full resolution processing**: No downsampling
- **Pure mathematical interpolation**: No LUT quantization artifacts
- **Float64 precision**: Maintained throughout entire pipeline  
- **Multi-threaded tiling**: Parallel processing of large images
- **Memory efficient**: Automatic chunking for large files

### Core Mathematical Constants
```python
_LOG65536 = np.log10(65536.0)  # ~4.816479930
```

This constant represents the density range mapping, where density values are normalized against the theoretical maximum density achievable with 16-bit precision.