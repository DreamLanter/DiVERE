"""智能裁剪布局管理器"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


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
        self.margin = 0.02  # 裁剪框之间的间距
        
    def get_layout_direction(self, image_aspect_ratio: float) -> str:
        """根据图片宽高比决定布局方向
        
        Args:
            image_aspect_ratio: 图片宽度/高度
            
        Returns:
            'horizontal' 或 'vertical'
        """
        if image_aspect_ratio > 1.2:  # 横图
            return 'horizontal'
        elif image_aspect_ratio < 0.8:  # 竖图
            return 'vertical'
        else:  # 接近正方形
            return 'horizontal'
    
    def find_next_position(self, 
                          existing_crops: List[Tuple[float, float, float, float]],
                          template_size: Optional[Tuple[float, float]] = None,
                          image_aspect_ratio: float = 1.0) -> Tuple[float, float, float, float]:
        """找到下一个裁剪的最佳位置
        
        Args:
            existing_crops: 现有裁剪列表 [(x, y, w, h), ...]
            template_size: 模板尺寸 (w, h)，如果为None则使用默认值
            image_aspect_ratio: 图片宽高比
            
        Returns:
            新裁剪位置 (x, y, w, h)
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
            # 使用默认尺寸
            if image_aspect_ratio > 1:  # 横图
                crop_w = self.default_crop_size
                crop_h = self.default_crop_size / image_aspect_ratio
            else:  # 竖图或正方形
                crop_w = self.default_crop_size * image_aspect_ratio
                crop_h = self.default_crop_size
            
            # 确保不超过最大尺寸
            crop_w = min(crop_w, self.max_crop_size)
            crop_h = min(crop_h, self.max_crop_size)
        
        # 获取布局方向
        layout_dir = self.get_layout_direction(image_aspect_ratio)
        
        # 如果没有现有裁剪，放在左上角
        if not existing_rects:
            return (self.margin, self.margin, crop_w, crop_h)
        
        # 查找合适的位置
        new_rect = self._find_position_by_direction(
            existing_rects, crop_w, crop_h, layout_dir
        )
        
        return new_rect.to_tuple()
    
    def _find_position_by_direction(self, 
                                   existing_rects: List[CropRect],
                                   crop_w: float, 
                                   crop_h: float,
                                   direction: str) -> CropRect:
        """根据方向查找位置"""
        
        if direction == 'horizontal':
            # 横向布局：从左到右，然后换行
            return self._find_horizontal_position(existing_rects, crop_w, crop_h)
        else:
            # 纵向布局：从上到下，然后换列
            return self._find_vertical_position(existing_rects, crop_w, crop_h)
    
    def _find_horizontal_position(self, 
                                 existing_rects: List[CropRect],
                                 crop_w: float, 
                                 crop_h: float) -> CropRect:
        """横向布局查找位置"""
        # 按行组织现有裁剪
        rows = self._group_by_rows(existing_rects)
        
        # 尝试在最后一行添加
        if rows:
            last_row = rows[-1]
            rightmost = max(last_row, key=lambda r: r.x + r.width)
            new_x = rightmost.x + rightmost.width + self.margin
            new_y = rightmost.y
            
            # 检查是否超出边界
            if new_x + crop_w <= 1.0 - self.margin:
                new_rect = CropRect(new_x, new_y, crop_w, crop_h)
                # 检查是否与其他裁剪重叠
                if not any(new_rect.overlaps_with(r) for r in existing_rects):
                    return new_rect
        
        # 需要新开一行
        if rows:
            bottommost = max(existing_rects, key=lambda r: r.y + r.height)
            new_y = bottommost.y + bottommost.height + self.margin
        else:
            new_y = self.margin
        
        new_x = self.margin
        
        # 确保不超出底部边界
        if new_y + crop_h > 1.0 - self.margin:
            # 如果超出，尝试缩小裁剪框
            crop_h = min(crop_h, 1.0 - self.margin - new_y)
            if crop_h < self.min_crop_size:
                # 如果太小，重置到左上角覆盖
                new_x, new_y = self.margin, self.margin
                crop_h = min(self.default_crop_size, 1.0 - 2 * self.margin)
        
        return CropRect(new_x, new_y, crop_w, crop_h)
    
    def _find_vertical_position(self, 
                               existing_rects: List[CropRect],
                               crop_w: float, 
                               crop_h: float) -> CropRect:
        """纵向布局查找位置"""
        # 按列组织现有裁剪
        columns = self._group_by_columns(existing_rects)
        
        # 尝试在最后一列添加
        if columns:
            last_column = columns[-1]
            bottommost = max(last_column, key=lambda r: r.y + r.height)
            new_x = bottommost.x
            new_y = bottommost.y + bottommost.height + self.margin
            
            # 检查是否超出边界
            if new_y + crop_h <= 1.0 - self.margin:
                new_rect = CropRect(new_x, new_y, crop_w, crop_h)
                # 检查是否与其他裁剪重叠
                if not any(new_rect.overlaps_with(r) for r in existing_rects):
                    return new_rect
        
        # 需要新开一列
        if columns:
            rightmost = max(existing_rects, key=lambda r: r.x + r.width)
            new_x = rightmost.x + rightmost.width + self.margin
        else:
            new_x = self.margin
        
        new_y = self.margin
        
        # 确保不超出右边界
        if new_x + crop_w > 1.0 - self.margin:
            # 如果超出，尝试缩小裁剪框
            crop_w = min(crop_w, 1.0 - self.margin - new_x)
            if crop_w < self.min_crop_size:
                # 如果太小，重置到左上角覆盖
                new_x, new_y = self.margin, self.margin
                crop_w = min(self.default_crop_size, 1.0 - 2 * self.margin)
        
        return CropRect(new_x, new_y, crop_w, crop_h)
    
    def _group_by_rows(self, rects: List[CropRect]) -> List[List[CropRect]]:
        """将裁剪框按行分组"""
        if not rects:
            return []
        
        # 按y坐标排序
        sorted_rects = sorted(rects, key=lambda r: r.y)
        rows = []
        current_row = [sorted_rects[0]]
        current_y = sorted_rects[0].y
        
        for rect in sorted_rects[1:]:
            # 如果y坐标相近，认为是同一行
            if abs(rect.y - current_y) < 0.05:
                current_row.append(rect)
            else:
                rows.append(current_row)
                current_row = [rect]
                current_y = rect.y
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _group_by_columns(self, rects: List[CropRect]) -> List[List[CropRect]]:
        """将裁剪框按列分组"""
        if not rects:
            return []
        
        # 按x坐标排序
        sorted_rects = sorted(rects, key=lambda r: r.x)
        columns = []
        current_column = [sorted_rects[0]]
        current_x = sorted_rects[0].x
        
        for rect in sorted_rects[1:]:
            # 如果x坐标相近，认为是同一列
            if abs(rect.x - current_x) < 0.05:
                current_column.append(rect)
            else:
                columns.append(current_column)
                current_column = [rect]
                current_x = rect.x
        
        if current_column:
            columns.append(current_column)
        
        return columns
