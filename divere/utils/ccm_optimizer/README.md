# CCM优化器

这是一个用于优化DiVERE管线参数的简单优化器，使用scipy.optimize来最小化ColorChecker 24色块的RGB MSE。

## 功能特性

- **参数优化**: 优化DiVERE管线的关键参数
  - `primaries_xy`: 输入色彩空间的RGB基色坐标
  - `gamma`: 密度反差参数
  - `dmax`: 最大密度参数
  - `r_gain`: R通道增益
  - `b_gain`: B通道增益

- **目标函数**: 最小化24个ColorChecker色块的RGB MSE
- **参考标准**: 使用ACEScg色彩空间的参考RGB值
- **边界约束**: 为每个参数设置合理的优化范围

## 文件结构

```
ccm_optimizer/
├── optimizer.py          # 核心优化器类
├── pipeline.py           # DiVERE管线模拟器
├── extractor.py          # 色块提取器
├── example_usage.py      # 使用示例
├── colorchecker_acescg_rgb_values.json  # 参考RGB值
└── README.md            # 说明文档
```

## 快速开始

### 1. 基本使用

```python
from optimizer import CCMOptimizer

# 创建优化器
optimizer = CCMOptimizer()

# 准备输入色块数据（从图像提取的24个色块RGB值）
input_patches = {
    'A1': (0.12, 0.08, 0.06),    # 深皮肤
    'A2': (0.45, 0.30, 0.23),    # 浅皮肤
    # ... 其他22个色块
    'D6': (0.03, 0.03, 0.03)     # 黑
}

# 执行优化
result = optimizer.optimize(
    input_patches,
    method='L-BFGS-B',
    max_iter=1000,
    tolerance=1e-6
)

# 查看结果
print(f"优化成功: {result['success']}")
print(f"最终MSE: {result['mse']:.6f}")
print(f"优化参数: {result['parameters']}")
```

### 2. 从图像直接优化

```python
from optimizer import optimize_from_image
import cv2

# 加载图像
image = cv2.imread('your_image.tif')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

# 定义ColorChecker角点坐标
corners = [
    (100, 100),   # 左上
    (700, 100),   # 右上
    (700, 500),   # 右下
    (100, 500)    # 左下
]

# 执行优化
result = optimize_from_image(
    image_rgb, corners,
    max_iter=1000,
    tolerance=1e-6
)
```

## 参数说明

### 优化参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `primaries_xy` | (3,2) | R:(0.5,0.9)×(0.1,0.4)<br>G:(0.1,0.4)×(0.6,0.9)<br>B:(0.1,0.4)×(0.0,0.3) | RGB基色xy坐标 |
| `gamma` | float | 1.5-4.0 | 密度反差参数 |
| `dmax` | float | 1.0-3.0 | 最大密度参数 |
| `r_gain` | float | -0.5-0.5 | R通道增益 |
| `b_gain` | float | -0.5-0.5 | B通道增益 |

### 优化选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `method` | 'L-BFGS-B' | 优化算法 |
| `max_iter` | 1000 | 最大迭代次数 |
| `tolerance` | 1e-6 | 收敛容差 |

## 优化算法

使用`scipy.optimize.minimize`的`L-BFGS-B`方法，这是一个基于梯度的优化算法，适合处理有边界约束的连续优化问题。

## 输出结果

优化器返回一个包含以下信息的字典：

```python
{
    'success': bool,           # 是否成功收敛
    'mse': float,             # 最终MSE值
    'iterations': int,         # 迭代次数
    'parameters': dict,        # 优化后的参数
    'raw_result': object,      # scipy优化结果对象
    'evaluation': dict         # 最终评估结果（如果使用optimize_from_image）
}
```

## 注意事项

1. **色块提取**: 确保从原图（不是preview）中提取色块
2. **采样边距**: 提取器默认使用中心30%区域，避免边缘效应
3. **色彩空间**: 输入图像应为0-1范围的RGB值
4. **参考标准**: 使用ACEScg色彩空间的参考值进行优化

## 运行示例

```bash
# 运行基本示例
python example_usage.py

# 运行优化器测试
python optimizer.py
```

## 依赖要求

- numpy
- scipy
- opencv-python (cv2)
- 其他DiVERE核心模块

## 故障排除

### 常见问题

1. **导入错误**: 确保在DiVERE项目根目录下运行
2. **色块提取失败**: 检查角点坐标是否正确
3. **优化不收敛**: 调整参数边界或增加迭代次数
4. **MSE过高**: 检查输入色块数据质量

### 调试模式

在`extractor.py`中设置`sample_margin=0.3`来调整采样区域大小，避免边缘效应。
