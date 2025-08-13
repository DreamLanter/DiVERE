#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCM (Color Correction Matrix) 优化器

基于ColorChecker色卡自动优化扫描仪色彩空间参数的工具包。

主要功能:
- 从原图提取ColorChecker色块
- 模拟DiVERE完整处理管线
- 优化输入色彩空间基色和处理参数
- 与标准ACEScg RGB值比较并最小化MSE损失

模块:
- extractor: 色块提取器
- pipeline: DiVERE管线模拟器
- optimizer: 核心优化器
- standalone_optimizer: 独立优化器
"""

__all__ = ['CCMOptimizer', 'optimize_from_image']
