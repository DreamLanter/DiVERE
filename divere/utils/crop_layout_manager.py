"""智能裁剪布局管理器"""

from typing import List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from divere.core.data_types import CropAddDirection, CropInstance
from .orientation_direction_mapper import convert_visual_to_standard_direction


@dataclass
class DirectionConfig:
    """简化的方向配置 - 仅定义核心移动方向"""
    primary_axis: str  # 'x' 或 'y' - 主要移动轴
    primary_positive: bool  # True表示正向移动，False表示负向移动
    secondary_axis: str  # 'x' 或 'y' - 次要移动轴（开新行/列时使用）
    secondary_positive: bool  # True表示正向移动，False表示负向移动


@dataclass
class CropRect:
    """裁剪矩形（归一化坐标）"""
    x: float
    y: float
    width: float
    height: float
    
    def overlaps_with(self, other: 'CropRect', margin: float = 0.01) -> bool:
        """检查是否与另一个矩形重叠（带边距）"""
        return not (
            self.x + self.width + margin <= other.x or
            other.x + other.width + margin <= self.x or
            self.y + self.height + margin <= other.y or
            other.y + other.height + margin <= self.y
        )
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """转换为(x, y, w, h)元组"""
        return (self.x, self.y, self.width, self.height)


class CropLayoutManager:
    """智能管理裁剪布局"""
    
    def __init__(self):
        self.min_crop_size = 0.1  # 最小裁剪尺寸（归一化）
        self.max_crop_size = 0.5  # 最大裁剪尺寸（归一化）
        self.default_crop_size = 0.25  # 默认裁剪尺寸（归一化）
        self.margin = 0.00  # 边界间距 可以允许没有边距。
        self.item_spacing_ratio = 0.0555  # 同行/同列内部间距比例（相对于crop长边）0.555
        self.row_spacing_ratio = 0.12  # 行与行之间间距比例（相对于crop高度）
        self.column_spacing_ratio = 0.12  # 列与列之间间距比例（相对于crop宽度）
        
        # 完整的8个方向配置映射
        # 简化的8种方向配置 - 仅定义核心移动方向
        self.direction_configs = {
            CropAddDirection.DOWN_RIGHT: DirectionConfig(
                primary_axis='y',           # 主要向下移动
                primary_positive=True,      # 向下（正向）
                secondary_axis='x',         # 次要向右移动
                secondary_positive=True     # 向右（正向）
            ),
            CropAddDirection.DOWN_LEFT: DirectionConfig(
                primary_axis='y',           # 主要向下移动
                primary_positive=True,      # 向下（正向）
                secondary_axis='x',         # 次要向左移动
                secondary_positive=False    # 向左（负向）
            ),
            CropAddDirection.RIGHT_DOWN: DirectionConfig(
                primary_axis='x',           # 主要向右移动
                primary_positive=True,      # 向右（正向）
                secondary_axis='y',         # 次要向下移动
                secondary_positive=True     # 向下（正向）
            ),
            CropAddDirection.RIGHT_UP: DirectionConfig(
                primary_axis='x',           # 主要向右移动
                primary_positive=True,      # 向右（正向）
                secondary_axis='y',         # 次要向上移动
                secondary_positive=False    # 向上（负向）
            ),
            CropAddDirection.UP_LEFT: DirectionConfig(
                primary_axis='y',           # 主要向上移动
                primary_positive=False,     # 向上（负向）
                secondary_axis='x',         # 次要向左移动
                secondary_positive=False    # 向左（负向）
            ),
            CropAddDirection.UP_RIGHT: DirectionConfig(
                primary_axis='y',           # 主要向上移动
                primary_positive=False,     # 向上（负向）
                secondary_axis='x',         # 次要向右移动
                secondary_positive=True     # 向右（正向）
            ),
            CropAddDirection.LEFT_UP: DirectionConfig(
                primary_axis='x',           # 主要向左移动
                primary_positive=False,     # 向左（负向）
                secondary_axis='y',         # 次要向上移动
                secondary_positive=False    # 向上（负向）
            ),
            CropAddDirection.LEFT_DOWN: DirectionConfig(
                primary_axis='x',           # 主要向左移动
                primary_positive=False,     # 向左（负向）
                secondary_axis='y',         # 次要向下移动
                secondary_positive=True     # 向下（正向）
            )
        }
        
    
    def _calculate_item_spacing(self, crop_w: float, crop_h: float, layout_direction: str) -> float:
        """根据crop尺寸和布局方向计算实际的item间距
        
        Args:
            crop_w: 裁剪框宽度
            crop_h: 裁剪框高度
            layout_direction: 布局方向 ('horizontal' 或 'vertical')
            
        Returns:
            实际的间距值（归一化坐标）
        """
        if layout_direction == 'horizontal':
            # 横向布局：间距相对于crop宽度（水平方向的长边）
            return crop_w * self.item_spacing_ratio
        else:
            # 纵向布局：间距相对于crop高度（垂直方向的长边）
            return crop_h * self.item_spacing_ratio
    
    def _calculate_row_spacing(self, crop_h: float) -> float:
        """根据crop高度计算行间距"""
        return crop_h * self.row_spacing_ratio
    
    def _calculate_column_spacing(self, crop_w: float) -> float:
        """根据crop宽度计算列间距"""
        return crop_w * self.column_spacing_ratio
    
    
    def find_next_position(self, 
                          existing_crops: List[Tuple[float, float, float, float]],
                          template_size: Optional[Tuple[float, float]] = None,
                          direction: CropAddDirection = CropAddDirection.DOWN_RIGHT,
                          orientation: int = 0) -> Tuple[float, float, float, float]:
        """找到下一个裁剪的最佳位置（支持orientation-aware坐标旋转）
        
        Args:
            existing_crops: 现有裁剪列表 [(x, y, w, h), ...]（归一化坐标）
            template_size: 模板尺寸 (w, h)，如果为None则使用最后一个裁剪的尺寸或默认值
            direction: 指定的添加方向（用户视觉方向，保持原始语义）
            orientation: 图像旋转角度 (0, 90, 180, 270)，用于坐标计算时的旋转变换
            
        Returns:
            新裁剪位置 (x, y, w, h)（归一化坐标）
            
        Note:
            direction参数表示用户的视觉意图，orientation会在坐标计算时应用旋转变换。
            支持8个方向：
            - DOWN_RIGHT (↓→): 优先向下，边缘时向右
            - DOWN_LEFT (↓←): 优先向下，边缘时向左
            - RIGHT_DOWN (→↓): 优先向右，边缘时向下  
            - RIGHT_UP (→↑): 优先向右，边缘时向上
            - UP_LEFT (↑←): 优先向上，边缘时向左
            - UP_RIGHT (↑→): 优先向上，边缘时向右
            - LEFT_UP (←↑): 优先向左，边缘时向上
            - LEFT_DOWN (←↓): 优先向左，边缘时向下
        """
        
        # 转换现有裁剪为CropRect对象
        existing_rects = [CropRect(x, y, w, h) for x, y, w, h in existing_crops]
        
        # 确定裁剪尺寸
        if template_size:
            crop_w, crop_h = template_size
        elif existing_rects:
            # 使用最后一个裁剪的尺寸
            last_crop = existing_rects[-1]
            crop_w, crop_h = last_crop.width, last_crop.height
        else:
            # 使用默认尺寸（正方形）
            crop_w = self.default_crop_size
            crop_h = self.default_crop_size
            
            # 确保不超过最大尺寸
            crop_w = min(crop_w, self.max_crop_size)
            crop_h = min(crop_h, self.max_crop_size)
        
        # 根据用户视觉方向和orientation查找合适的位置
        # 即使existing_rects为空，也要通过_find_position处理方向转换
        new_rect = self._find_position(
            existing_rects, crop_w, crop_h, direction, orientation
        )
        
        return new_rect.to_tuple()
    
    def find_next_position_from_crops(self,
                                     existing_crops: List[CropInstance],
                                     template_size: Optional[Tuple[float, float]] = None,
                                     direction: CropAddDirection = CropAddDirection.DOWN_RIGHT,
                                     orientation: int = 0) -> CropInstance:
        """从CropInstance列表中找到下一个裁剪位置（preset驱动版本）
        
        Args:
            existing_crops: 现有CropInstance列表（原图坐标系）
            template_size: 模板尺寸 (w, h)
            direction: 用户视觉方向
            orientation: 当前显示orientation
            
        Returns:
            新的CropInstance对象（原图坐标系）
            
        Note:
            这个方法专门用于preset驱动的crop系统，确保所有数据都基于原图坐标系
        """
        # 转换CropInstance列表为元组格式
        existing_tuples = [crop.rect_norm for crop in existing_crops]
        
        # 使用现有的find_next_position方法计算位置
        new_rect_tuple = self.find_next_position(
            existing_crops=existing_tuples,
            template_size=template_size,
            direction=direction,
            orientation=orientation
        )
        
        # 生成唯一ID
        import uuid
        crop_id = f"crop_{len(existing_crops) + 1}_{uuid.uuid4().hex[:8]}"
        
        # 创建新的CropInstance（注意：orientation为0，因为坐标已经是原图坐标系）
        return CropInstance(
            id=crop_id,
            name=f"裁剪 {len(existing_crops) + 1}",
            rect_norm=new_rect_tuple,
            orientation=0  # 始终为0，因为rect_norm已经是原图坐标系
        )
    
    def _find_position(self, 
                       existing_rects: List[CropRect],
                       crop_w: float, 
                       crop_h: float,
                       direction: CropAddDirection,
                       orientation: int) -> CropRect:
        """简化的位置查找函数 - 仅基于最后一个crop的位置"""
        if not existing_rects:
            # 第一个crop：根据方向配置决定初始位置
            actual_direction = convert_visual_to_standard_direction(direction, orientation)
            config = self.direction_configs[actual_direction]
            return self._get_initial_position(crop_w, crop_h, config)
        
        # 使用orientation_direction_mapper转换方向
        actual_direction = convert_visual_to_standard_direction(direction, orientation)
        config = self.direction_configs[actual_direction]
        
        # 获取最后一个crop作为参考
        last_crop = existing_rects[-1]
        
        # 计算间距
        spacing = self._calculate_spacing(crop_w, crop_h, config)
        
        # 检查主方向是否有空间
        if self._can_add_primary(last_crop, config, crop_w, crop_h, spacing):
            # 在主方向添加
            return self._calculate_primary_position(last_crop, config, crop_w, crop_h, spacing)
        else:
            # 在次方向开新行/列
            return self._calculate_secondary_position(last_crop, config, crop_w, crop_h)
    
    def _calculate_spacing(self, crop_w: float, crop_h: float, config: DirectionConfig) -> float:
        """计算crop间距"""
        # 根据主轴方向决定间距计算基准
        if config.primary_axis == 'x':
            return crop_w * self.item_spacing_ratio
        else:
            return crop_h * self.item_spacing_ratio
    
    def _can_add_primary(self, last_crop: CropRect, config: DirectionConfig, 
                        crop_w: float, crop_h: float, spacing: float) -> bool:
        """检查主方向是否有足够空间"""
        spacing = - spacing/2 # 在判定是否还有剩余空间时2，减少间距的一半，允许更紧凑的布局
        if config.primary_axis == 'x':
            if config.primary_positive:
                # 向右：检查右边界
                next_x = last_crop.x + last_crop.width + spacing
                return (next_x + crop_w) <= (1.0 - self.margin)
            else:
                # 向左：检查左边界
                next_x = last_crop.x - spacing - crop_w
                return next_x >= self.margin
        else:  # primary_axis == 'y'
            if config.primary_positive:
                # 向下：检查底边界
                next_y = last_crop.y + last_crop.height + spacing
                return (next_y + crop_h) <= (1.0 - self.margin)
            else:
                # 向上：检查顶边界
                next_y = last_crop.y - spacing - crop_h
                return next_y >= self.margin
    
    def _calculate_primary_position(self, last_crop: CropRect, config: DirectionConfig,
                                   crop_w: float, crop_h: float, spacing: float) -> CropRect:
        """在主方向计算新crop位置（紧贴最后一个crop）"""
        if config.primary_axis == 'x':
            if config.primary_positive:
                # 向右
                new_x = last_crop.x + last_crop.width + spacing
                new_y = last_crop.y  # 保持同一行
            else:
                # 向左
                new_x = last_crop.x - spacing - crop_w
                new_y = last_crop.y  # 保持同一行
        else:  # primary_axis == 'y'
            if config.primary_positive:
                # 向下
                new_x = last_crop.x  # 保持同一列
                new_y = last_crop.y + last_crop.height + spacing
            else:
                # 向上
                new_x = last_crop.x  # 保持同一列
                new_y = last_crop.y - spacing - crop_h
        
        return CropRect(new_x, new_y, crop_w, crop_h)
    
    def _calculate_secondary_position(self, last_crop: CropRect, config: DirectionConfig,
                                     crop_w: float, crop_h: float) -> CropRect:
        """在次方向开新行/列（基于最后一个crop的位置）"""
        # 计算次方向的间距
        if config.secondary_axis == 'x':
            secondary_spacing = crop_w * self.column_spacing_ratio
        else:
            secondary_spacing = crop_h * self.row_spacing_ratio
        
        if config.secondary_axis == 'x':
            if config.secondary_positive:
                # 向右开新列
                new_x = last_crop.x + last_crop.width + secondary_spacing
                # Y坐标：根据主方向决定对齐方式
                if config.primary_positive:  # DOWN方向：从顶部开始
                    new_y = self.margin
                else:  # UP方向：从底部开始
                    new_y = 1.0 - self.margin - crop_h
            else:
                # 向左开新列
                new_x = last_crop.x - secondary_spacing - crop_w
                # Y坐标：根据主方向决定对齐方式
                if config.primary_positive:  # DOWN方向：从顶部开始
                    new_y = self.margin
                else:  # UP方向：从底部开始
                    new_y = 1.0 - self.margin - crop_h
        else:  # secondary_axis == 'y'
            if config.secondary_positive:
                # 向下开新行
                new_y = last_crop.y + last_crop.height + secondary_spacing
                # X坐标：根据主方向决定对齐方式
                if config.primary_positive:  # RIGHT方向：从左边开始
                    new_x = self.margin
                else:  # LEFT方向：从右边开始
                    new_x = 1.0 - self.margin - crop_w
            else:
                # 向上开新行
                new_y = last_crop.y - secondary_spacing - crop_h
                # X坐标：根据主方向决定对齐方式
                if config.primary_positive:  # RIGHT方向：从左边开始
                    new_x = self.margin
                else:  # LEFT方向：从右边开始
                    new_x = 1.0 - self.margin - crop_w
        
        # 边界检查和修正
        new_x = max(self.margin, min(new_x, 1.0 - self.margin - crop_w))
        new_y = max(self.margin, min(new_y, 1.0 - self.margin - crop_h))
        
        return CropRect(new_x, new_y, crop_w, crop_h)
    
    def _get_initial_position(self, crop_w: float, crop_h: float, config: DirectionConfig) -> CropRect:
        """根据方向配置确定第一个crop的初始位置"""
        # 根据主方向和次方向确定合适的起始位置
        if config.primary_axis == 'y':  # 主要是纵向移动
            if config.primary_positive:  # 向下
                if config.secondary_positive:  # 向右
                    # DOWN_RIGHT: 从左上角开始
                    x, y = self.margin, self.margin
                else:  # 向左  
                    # DOWN_LEFT: 从右上角开始
                    x, y = 1.0 - self.margin - crop_w, self.margin
            else:  # 向上
                if config.secondary_positive:  # 向右
                    # UP_RIGHT: 从左下角开始
                    x, y = self.margin, 1.0 - self.margin - crop_h
                else:  # 向左
                    # UP_LEFT: 从右下角开始
                    x, y = 1.0 - self.margin - crop_w, 1.0 - self.margin - crop_h
        else:  # config.primary_axis == 'x', 主要是横向移动
            if config.primary_positive:  # 向右
                if config.secondary_positive:  # 向下
                    # RIGHT_DOWN: 从左上角开始
                    x, y = self.margin, self.margin
                else:  # 向上
                    # RIGHT_UP: 从左下角开始
                    x, y = self.margin, 1.0 - self.margin - crop_h
            else:  # 向左
                if config.secondary_positive:  # 向下
                    # LEFT_DOWN: 从右上角开始
                    x, y = 1.0 - self.margin - crop_w, self.margin
                else:  # 向上
                    # LEFT_UP: 从右下角开始
                    x, y = 1.0 - self.margin - crop_w, 1.0 - self.margin - crop_h
        
        return CropRect(x, y, crop_w, crop_h)


class PresetCropManager:
    """Preset驱动的Crop管理器
    
    负责管理基于preset的crop操作，确保所有数据都基于原图坐标系存储
    """
    
    def __init__(self):
        self.layout_manager = CropLayoutManager()
    
    def add_crop_to_preset(self, 
                          preset: 'Preset',  # type: ignore
                          direction: CropAddDirection,
                          display_orientation: int = 0,
                          template_size: Optional[Tuple[float, float]] = None) -> 'CropInstance':  # type: ignore
        """向preset添加新的crop
        
        Args:
            preset: 目标preset对象
            direction: 用户选择的视觉方向
            display_orientation: 当前UI显示的orientation
            template_size: 可选的模板尺寸
            
        Returns:
            新创建的CropInstance对象（原图坐标系）
        """
        # 获取现有的crops
        existing_crops = preset.get_crop_instances()
        
        # 使用layout_manager计算新位置
        new_crop = self.layout_manager.find_next_position_from_crops(
            existing_crops=existing_crops,
            template_size=template_size,
            direction=direction,
            orientation=display_orientation
        )
        
        # 更新preset中的crops数据
        updated_crops = existing_crops + [new_crop]
        preset.set_crop_instances(updated_crops, new_crop.id)
        
        return new_crop
    
    def remove_crop_from_preset(self, preset: 'Preset', crop_id: str) -> bool:  # type: ignore
        """从preset中移除指定的crop
        
        Args:
            preset: 目标preset对象
            crop_id: 要移除的crop ID
            
        Returns:
            是否成功移除
        """
        existing_crops = preset.get_crop_instances()
        updated_crops = [crop for crop in existing_crops if crop.id != crop_id]
        
        if len(updated_crops) == len(existing_crops):
            return False  # 没有找到要删除的crop
            
        # 如果删除的是active crop，需要选择新的active crop
        new_active_id = None
        if updated_crops:
            if preset.active_crop_id == crop_id:
                new_active_id = updated_crops[0].id
            else:
                new_active_id = preset.active_crop_id
                
        preset.set_crop_instances(updated_crops, new_active_id)
        return True
    
    def get_display_crops(self, 
                         preset: 'Preset',  # type: ignore
                         display_orientation: int) -> List[Tuple[float, float, float, float]]:
        """获取用于UI显示的crop坐标列表
        
        Args:
            preset: preset对象
            display_orientation: 当前显示orientation
            
        Returns:
            转换为当前显示方向的crop坐标列表
        """
        crops = preset.get_crop_instances()
        display_crops = []
        
        for crop in crops:
            # 将原图坐标转换为显示坐标
            display_rect = self.transform_crop_for_display(
                crop.rect_norm, 
                display_orientation
            )
            display_crops.append(display_rect)
            
        return display_crops
    
    def transform_crop_for_display(self, 
                                  original_rect: Tuple[float, float, float, float],
                                  display_orientation: int) -> Tuple[float, float, float, float]:
        """将原图坐标系的crop转换为显示坐标系
        
        Args:
            original_rect: 原图坐标系中的crop矩形 (x, y, w, h)
            display_orientation: 显示orientation (0, 90, 180, 270)
            
        Returns:
            显示坐标系中的crop矩形
        """
        if display_orientation == 0:
            return original_rect
            
        x, y, w, h = original_rect
        
        # 根据orientation转换坐标
        if display_orientation == 90:  # CCW 90°
            # (x,y) → (1-y-h, x)
            # (w,h) → (h, w)
            return (1.0 - y - h, x, h, w)
        elif display_orientation == 180:  # 180°
            # (x,y) → (1-x-w, 1-y-h)
            # (w,h) → (w, h)
            return (1.0 - x - w, 1.0 - y - h, w, h)
        elif display_orientation == 270:  # CCW 270°
            # (x,y) → (y, 1-x-w)
            # (w,h) → (h, w)  
            return (y, 1.0 - x - w, h, w)
        else:
            raise ValueError(f"Unsupported display_orientation: {display_orientation}")
    
    def save_preset(self, preset: 'Preset', file_path: str):  # type: ignore
        """保存preset到文件（简化版本，实际实现需要根据项目的保存逻辑）
        
        Args:
            preset: preset对象
            file_path: 保存路径
        """
        # 这里应该调用项目中实际的preset保存逻辑
        # 比如通过PresetManager来保存
        import json
        with open(file_path, 'w') as f:
            json.dump(preset.to_dict(), f, indent=2)
