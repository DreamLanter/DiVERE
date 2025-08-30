"""
GPU加速器模块 - 跨平台GPU加速支持
支持OpenCL, CUDA, Metal等多种GPU计算后端
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
from abc import ABC, abstractmethod

# GPU库导入（可选）
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    import Metal
    import MetalPerformanceShaders as MPS
    import objc
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


class GPUComputeEngine(ABC):
    """GPU计算引擎抽象基类"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查计算引擎是否可用"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        pass
    
    @abstractmethod
    def density_inversion_gpu(self, image: np.ndarray, gamma: float, 
                             dmax: float, pivot: float) -> np.ndarray:
        """GPU加速的密度反相"""
        pass
    
    @abstractmethod
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        """GPU加速的曲线处理"""
        pass


class OpenCLEngine(GPUComputeEngine):
    """OpenCL计算引擎 - 跨平台支持"""
    
    def __init__(self):
        self.context = None
        self.queue = None
        self.device = None
        self.program = None
        self._initialize()
    
    def _initialize(self):
        """初始化OpenCL环境"""
        if not OPENCL_AVAILABLE:
            return
        
        try:
            # 寻找最佳GPU设备
            platforms = cl.get_platforms()
            best_device = None
            best_compute_units = 0
            
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    if device.type & cl.device_type.GPU:
                        compute_units = device.max_compute_units
                        if compute_units > best_compute_units:
                            best_device = device
                            best_compute_units = compute_units
            
            if best_device:
                self.device = best_device
                self.context = cl.Context([best_device])
                self.queue = cl.CommandQueue(self.context)
                self._build_kernels()
                
        except Exception as e:
            print(f"OpenCL初始化失败: {e}")
    
    def _build_kernels(self):
        """编译OpenCL内核"""
        kernel_source = '''
        __kernel void density_inversion(__global const float* input,
                                       __global float* output,
                                       const float gamma,
                                       const float dmax,
                                       const float pivot,
                                       const int size)
        {
            int i = get_global_id(0);
            if (i >= size) return;
            
            // 避免log(0)
            float safe_val = fmax(input[i], 1e-10f);
            
            // 密度反相计算
            float original_density = -log10(safe_val);
            float adjusted_density = pivot + (original_density - pivot) * gamma - dmax;
            
            // 转回线性空间
            output[i] = pow(10.0f, adjusted_density);
        }
        
        __kernel void curve_lut_apply(__global const float* input,
                                     __global float* output,
                                     __global const float* lut,
                                     const int image_size,
                                     const int lut_size)
        {
            int i = get_global_id(0);
            if (i >= image_size) return;
            
            // 归一化到LUT索引范围
            float normalized = 1.0f - clamp(input[i] * 0.000152587890625f, 0.0f, 1.0f);
            float index_f = normalized * (lut_size - 1);
            int index = (int)index_f;
            
            // 简单索引（可以改为插值）
            index = clamp(index, 0, lut_size - 1);
            output[i] = lut[index];
        }
        '''
        
        try:
            self.program = cl.Program(self.context, kernel_source).build()
        except Exception as e:
            print(f"OpenCL内核编译失败: {e}")
            self.program = None
    
    def is_available(self) -> bool:
        """检查OpenCL是否可用"""
        return (OPENCL_AVAILABLE and 
                self.context is not None and 
                self.queue is not None and 
                self.program is not None)
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "name": self.device.name,
            "type": "OpenCL",
            "compute_units": self.device.max_compute_units,
            "global_memory_mb": self.device.global_mem_size // 1024 // 1024,
            "max_work_group_size": self.device.max_work_group_size
        }
    
    def density_inversion_gpu(self, image: np.ndarray, gamma: float, 
                             dmax: float, pivot: float) -> np.ndarray:
        """GPU加速的密度反相"""
        if not self.is_available():
            raise RuntimeError("OpenCL不可用")
        
        # 展平数组以简化处理
        original_shape = image.shape
        image_flat = image.flatten().astype(np.float32)
        output_flat = np.zeros_like(image_flat)
        
        # 创建OpenCL缓冲区
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                             hostbuf=image_flat)
        output_buf = cl.Buffer(self.context, mf.WRITE_ONLY, image_flat.nbytes)
        
        # 执行内核
        global_size = (len(image_flat),)
        self.program.density_inversion(
            self.queue, global_size, None,
            input_buf, output_buf,
            np.float32(gamma), np.float32(dmax), np.float32(pivot),
            np.int32(len(image_flat))
        )
        
        # 读取结果
        cl.enqueue_copy(self.queue, output_flat, output_buf)
        
        # 恢复原始形状
        return output_flat.reshape(original_shape)
    
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        """GPU加速的曲线处理"""
        if not self.is_available():
            raise RuntimeError("OpenCL不可用")
        
        # 展平数组
        original_shape = density_array.shape
        density_flat = density_array.flatten().astype(np.float32)
        output_flat = np.zeros_like(density_flat)
        lut_float = lut.astype(np.float32)
        
        # 创建OpenCL缓冲区
        mf = cl.mem_flags
        input_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                             hostbuf=density_flat)
        output_buf = cl.Buffer(self.context, mf.WRITE_ONLY, density_flat.nbytes)
        lut_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                           hostbuf=lut_float)
        
        # 执行内核
        global_size = (len(density_flat),)
        self.program.curve_lut_apply(
            self.queue, global_size, None,
            input_buf, output_buf, lut_buf,
            np.int32(len(density_flat)), np.int32(len(lut_float))
        )
        
        # 读取结果
        cl.enqueue_copy(self.queue, output_flat, output_buf)
        
        # 恢复原始形状
        return output_flat.reshape(original_shape)


class CUDAEngine(GPUComputeEngine):
    """CUDA计算引擎 - NVIDIA GPU专用"""
    
    def __init__(self):
        self._available = CUDA_AVAILABLE
        if self._available:
            try:
                # 测试CUDA可用性
                cp.cuda.Device(0).use()
            except:
                self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def get_device_info(self) -> Dict[str, Any]:
        if not self.is_available():
            return {"available": False}
        
        device = cp.cuda.Device()
        return {
            "available": True,
            "name": device.attributes["Name"],
            "type": "CUDA",
            "compute_capability": device.compute_capability,
            "memory_mb": device.mem_info[1] // 1024 // 1024
        }
    
    def density_inversion_gpu(self, image: np.ndarray, gamma: float, 
                             dmax: float, pivot: float) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("CUDA不可用")
        
        # 转移到GPU
        image_gpu = cp.asarray(image)
        
        # 避免log(0)
        safe_img = cp.maximum(image_gpu, 1e-10)
        
        # 密度反相计算
        original_density = -cp.log10(safe_img)
        adjusted_density = pivot + (original_density - pivot) * gamma - dmax
        result_gpu = cp.power(10.0, adjusted_density)
        
        # 转回CPU
        return cp.asnumpy(result_gpu)
    
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("CUDA不可用")
        
        # 转移到GPU
        density_gpu = cp.asarray(density_array)
        lut_gpu = cp.asarray(lut)
        
        # 这里需要实现CUDA版本的LUT查表
        # 简化版本：使用CuPy的interp
        # 实际实现需要优化的索引操作
        
        # 转回CPU（占位实现）
        return cp.asnumpy(density_gpu)


class MetalEngine(GPUComputeEngine):
    """Metal计算引擎 - macOS原生最优性能"""
    
    def __init__(self):
        self.device = None
        self.command_queue = None
        self.library = None
        self._initialize()
    
    def _initialize(self):
        """初始化Metal环境"""
        if not METAL_AVAILABLE:
            return
        
        try:
            # 创建Metal设备和命令队列
            self.device = Metal.MTLCreateSystemDefaultDevice()
            if self.device:
                self.command_queue = self.device.newCommandQueue()
                self._create_compute_library()
        except Exception as e:
            print(f"Metal初始化失败: {e}")
    
    def _create_compute_library(self):
        """创建Metal计算着色器库"""
        # Metal着色器源代码
        metal_source = '''
        #include <metal_stdlib>
        using namespace metal;
        
        // 密度反相计算着色器（高精度版本）
        kernel void density_inversion(device const float* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     constant float& gamma [[buffer(2)]],
                                     constant float& dmax [[buffer(3)]],
                                     constant float& pivot [[buffer(4)]],
                                     uint index [[thread_position_in_grid]])
        {
            // 确保与CPU版本相同的精度处理
            float safe_val = max(input[index], 1e-10f);
            
            // 密度反相计算 - 使用高精度数学函数
            float original_density = -log10(safe_val);
            float adjusted_density = pivot + (original_density - pivot) * gamma - dmax;
            
            // 转回线性空间 - 使用precise标志确保精度
            output[index] = precise::pow(10.0f, adjusted_density);
        }
        
        // LUT查表着色器
        kernel void lut_apply(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             device const float* lut [[buffer(2)]],
                             constant uint& lut_size [[buffer(3)]],
                             uint index [[thread_position_in_grid]])
        {
            // 归一化到LUT索引范围
            float normalized = 1.0f - clamp(input[index] * 0.000152587890625f, 0.0f, 1.0f);
            float index_f = normalized * (lut_size - 1);
            uint lut_index = uint(index_f);
            
            // 边界检查
            lut_index = min(lut_index, lut_size - 1);
            output[index] = lut[lut_index];
        }
        
        // 矩阵乘法着色器（用于色彩空间转换）
        kernel void matrix_multiply_3x3(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       device const float* matrix [[buffer(2)]],
                                       uint index [[thread_position_in_grid]])
        {
            uint pixel_index = index / 3;
            uint component = index % 3;
            uint base_index = pixel_index * 3;
            
            float result = 0.0f;
            for (uint i = 0; i < 3; i++) {
                result += matrix[component * 3 + i] * input[base_index + i];
            }
            output[index] = result;
        }
        '''
        
        try:
            # 编译Metal着色器
            library = self.device.newLibraryWithSource_options_error_(
                metal_source, None, None
            )
            if library[0]:
                self.library = library[0]
            else:
                print(f"Metal着色器编译失败: {library[1]}")
        except Exception as e:
            print(f"Metal着色器创建失败: {e}")
    
    def is_available(self) -> bool:
        """检查Metal是否可用"""
        return (METAL_AVAILABLE and 
                self.device is not None and 
                self.command_queue is not None and 
                self.library is not None)
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if not self.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "name": self.device.name(),
            "type": "Metal",
            "unified_memory": self.device.hasUnifiedMemory(),
            "max_buffer_mb": self.device.maxBufferLength() // 1024 // 1024,
            "low_power": self.device.isLowPower()
        }
    
    def density_inversion_gpu(self, image: np.ndarray, gamma: float, 
                             dmax: float, pivot: float) -> np.ndarray:
        """Metal加速的密度反相"""
        if not self.is_available():
            raise RuntimeError("Metal不可用")
        
        # 展平数组
        original_shape = image.shape
        image_flat = image.flatten().astype(np.float32)
        output_flat = np.zeros_like(image_flat)
        
        # 创建Metal缓冲区
        input_buffer = self.device.newBufferWithBytes_length_options_(
            image_flat.tobytes(), 
            len(image_flat) * 4,  # float32 = 4 bytes
            Metal.MTLResourceStorageModeShared
        )
        
        output_buffer = self.device.newBufferWithLength_options_(
            len(image_flat) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        # 参数缓冲区
        gamma_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([gamma], dtype=np.float32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        dmax_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([dmax], dtype=np.float32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        pivot_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([pivot], dtype=np.float32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        # 创建计算管线
        function = self.library.newFunctionWithName_("density_inversion")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        # 创建命令缓冲区和编码器
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        # 设置管线和缓冲区
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(gamma_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(dmax_buffer, 0, 3)
        compute_encoder.setBuffer_offset_atIndex_(pivot_buffer, 0, 4)
        
        # 计算线程网格
        threads_per_threadgroup = Metal.MTLSize(256, 1, 1)
        threadgroups = Metal.MTLSize(
            (len(image_flat) + 255) // 256, 1, 1
        )
        
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # 读取结果 - 使用正确的Metal缓冲区访问方法
        result_ptr = output_buffer.contents()
        # 创建numpy数组直接指向Metal缓冲区内存
        result_array = np.frombuffer(
            result_ptr.as_buffer(len(image_flat) * 4), 
            dtype=np.float32
        ).copy()  # copy确保数据独立
        
        return result_array.reshape(original_shape)
    
    def curve_processing_gpu(self, density_array: np.ndarray, 
                           lut: np.ndarray) -> np.ndarray:
        """Metal加速的曲线处理"""
        if not self.is_available():
            raise RuntimeError("Metal不可用")
        
        # 展平数组
        original_shape = density_array.shape
        density_flat = density_array.flatten().astype(np.float32)
        output_flat = np.zeros_like(density_flat)
        lut_float = lut.astype(np.float32)
        
        # 创建Metal缓冲区
        input_buffer = self.device.newBufferWithBytes_length_options_(
            density_flat.tobytes(), 
            len(density_flat) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        output_buffer = self.device.newBufferWithLength_options_(
            len(density_flat) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        lut_buffer = self.device.newBufferWithBytes_length_options_(
            lut_float.tobytes(),
            len(lut_float) * 4,
            Metal.MTLResourceStorageModeShared
        )
        
        lut_size_buffer = self.device.newBufferWithBytes_length_options_(
            np.array([len(lut_float)], dtype=np.uint32).tobytes(), 4,
            Metal.MTLResourceStorageModeShared
        )
        
        # 创建计算管线
        function = self.library.newFunctionWithName_("lut_apply")
        pipeline_state = self.device.newComputePipelineStateWithFunction_error_(function, None)[0]
        
        # 创建命令缓冲区和编码器
        command_buffer = self.command_queue.commandBuffer()
        compute_encoder = command_buffer.computeCommandEncoder()
        
        # 设置管线和缓冲区
        compute_encoder.setComputePipelineState_(pipeline_state)
        compute_encoder.setBuffer_offset_atIndex_(input_buffer, 0, 0)
        compute_encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
        compute_encoder.setBuffer_offset_atIndex_(lut_buffer, 0, 2)
        compute_encoder.setBuffer_offset_atIndex_(lut_size_buffer, 0, 3)
        
        # 计算线程网格
        threads_per_threadgroup = Metal.MTLSize(256, 1, 1)
        threadgroups = Metal.MTLSize(
            (len(density_flat) + 255) // 256, 1, 1
        )
        
        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            threadgroups, threads_per_threadgroup
        )
        
        compute_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # 读取结果 - 使用正确的Metal缓冲区访问方法
        result_ptr = output_buffer.contents()
        # 创建numpy数组直接指向Metal缓冲区内存
        result_array = np.frombuffer(
            result_ptr.as_buffer(len(density_flat) * 4), 
            dtype=np.float32
        ).copy()  # copy确保数据独立
        
        return result_array.reshape(original_shape)


class GPUAccelerator:
    """GPU加速器 - 统一的GPU加速接口"""
    
    def __init__(self):
        self.engines = []
        self.active_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """初始化所有可用的计算引擎"""
        # 优先级顺序：Metal > OpenCL > CUDA（macOS原生优先）
        engines_to_try = [
            ("Metal", MetalEngine),
            ("OpenCL", OpenCLEngine), 
            ("CUDA", CUDAEngine),
        ]
        
        for name, engine_class in engines_to_try:
            try:
                engine = engine_class()
                if engine.is_available():
                    self.engines.append((name, engine))
                    if self.active_engine is None:
                        self.active_engine = engine
                        print(f"🚀 使用GPU引擎: {name}")
            except Exception as e:
                print(f"⚠️  {name}引擎初始化失败: {e}")
    
    def is_available(self) -> bool:
        """检查是否有可用的GPU加速"""
        return self.active_engine is not None
    
    def get_available_engines(self) -> List[str]:
        """获取所有可用的引擎列表"""
        return [name for name, engine in self.engines]
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取当前激活设备的信息"""
        if not self.is_available():
            return {"available": False, "fallback": "CPU"}
        
        info = self.active_engine.get_device_info()
        info["engines_available"] = self.get_available_engines()
        return info
    
    def set_active_engine(self, engine_name: str) -> bool:
        """切换激活的计算引擎"""
        for name, engine in self.engines:
            if name == engine_name:
                self.active_engine = engine
                print(f"🔄 切换到GPU引擎: {engine_name}")
                return True
        return False
    
    def density_inversion_accelerated(self, image: np.ndarray, gamma: float, 
                                     dmax: float, pivot: float) -> np.ndarray:
        """GPU加速的密度反相，自动回退到CPU"""
        if not self.is_available():
            # CPU回退版本
            return self._density_inversion_cpu(image, gamma, dmax, pivot)
        
        try:
            return self.active_engine.density_inversion_gpu(image, gamma, dmax, pivot)
        except Exception as e:
            print(f"⚠️  GPU加速失败，回退到CPU: {e}")
            return self._density_inversion_cpu(image, gamma, dmax, pivot)
    
    def curve_processing_accelerated(self, density_array: np.ndarray, 
                                   lut: np.ndarray) -> np.ndarray:
        """GPU加速的曲线处理，自动回退到CPU"""
        if not self.is_available():
            return self._curve_processing_cpu(density_array, lut)
        
        try:
            return self.active_engine.curve_processing_gpu(density_array, lut)
        except Exception as e:
            print(f"⚠️  GPU加速失败，回退到CPU: {e}")
            return self._curve_processing_cpu(density_array, lut)
    
    def _density_inversion_cpu(self, image: np.ndarray, gamma: float, 
                              dmax: float, pivot: float) -> np.ndarray:
        """CPU版本的密度反相（回退方案）"""
        safe_img = np.maximum(image, 1e-10)
        original_density = -np.log10(safe_img)
        adjusted_density = pivot + (original_density - pivot) * gamma - dmax
        return np.power(10.0, adjusted_density)
    
    def _curve_processing_cpu(self, density_array: np.ndarray, 
                             lut: np.ndarray) -> np.ndarray:
        """CPU版本的曲线处理（回退方案）- 使用高精度插值"""
        # 高精度线性插值实现，避免简单索引造成的量化误差
        inv_range = 1.0 / 6.5536  # LOG65536的倒数
        normalized = 1.0 - np.clip(density_array * inv_range, 0.0, 1.0)
        
        # 使用NumPy的高精度插值而不是简单索引
        lut_indices = np.linspace(0.0, 1.0, len(lut), dtype=np.float64)
        result = np.interp(normalized.flatten(), lut_indices, lut).astype(density_array.dtype)
        
        return result.reshape(density_array.shape)


# 全局GPU加速器实例
_gpu_accelerator = None

def get_gpu_accelerator() -> GPUAccelerator:
    """获取全局GPU加速器实例"""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator
