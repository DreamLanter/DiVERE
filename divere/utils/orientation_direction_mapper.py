"""
Orientation-aware direction mapping utility

This module provides utilities to convert visual (display-oriented) crop add directions
to standard coordinate system directions, taking into account image orientation.
"""

from typing import Dict, List
from divere.core.data_types import CropAddDirection


class OrientationDirectionMapper:
    """
    Maps visual crop add directions to standard coordinate system directions
    based on image orientation.
    
    Visual directions are what the user sees on screen (e.g., "down-right" means
    down and right as displayed). Standard directions are in the original image
    coordinate system (0° orientation).
    """
    
    # Direction enumeration order for mapping (must match CropAddDirection order)
    _DIRECTION_ORDER = [
        CropAddDirection.DOWN_RIGHT,   # 0
        CropAddDirection.DOWN_LEFT,    # 1
        CropAddDirection.RIGHT_DOWN,   # 2
        CropAddDirection.RIGHT_UP,     # 3
        CropAddDirection.UP_LEFT,      # 4
        CropAddDirection.UP_RIGHT,     # 5
        CropAddDirection.LEFT_UP,      # 6
        CropAddDirection.LEFT_DOWN,    # 7
    ]
    
    # Direction mapping matrix: [original_direction_index][orientation_quarter] = target_direction_index
    # orientation_quarter = (orientation // 90) % 4
    _DIRECTION_MAPPING = [
        # DOWN_RIGHT (0): 向下优先，边缘向右
        [0, 7, 4, 3],  # 0°→DOWN_RIGHT, 90°→LEFT_UP, 180°→UP_LEFT, 270°→RIGHT_DOWN

        # DOWN_LEFT (1): 向下优先，边缘向左
        [1, 6, 5, 2],  # 0°→DOWN_LEFT, 90°→LEFT_DOWN, 180°→UP_RIGHT, 270°→RIGHT_UP

        # RIGHT_DOWN (2): 向右优先，边缘向下
        [2, 1, 6, 5],  # 0°→RIGHT_DOWN, 90°→DOWN_RIGHT, 180°→LEFT_UP, 270°→UP_LEFT

        # RIGHT_UP (3): 向右优先，边缘向上
        [3, 0, 7, 4],  # 0°→RIGHT_UP, 90°→DOWN_LEFT, 180°→LEFT_DOWN, 270°→UP_RIGHT

        # UP_LEFT (4): 向上优先，边缘向左
        [4, 2, 0, 7],  # 0°→UP_LEFT, 90°→RIGHT_DOWN, 180°→DOWN_RIGHT, 270°→LEFT_UP

        # UP_RIGHT (5): 向上优先，边缘向右
        [5, 2, 1, 6],  # 0°→UP_RIGHT, 90°→RIGHT_UP, 180°→DOWN_LEFT, 270°→LEFT_DOWN

        # LEFT_UP (6): 向左优先，边缘向上
        [6, 5, 2, 1],  # 0°→LEFT_UP, 90°→UP_LEFT, 180°→RIGHT_DOWN, 270°→DOWN_RIGHT

        # LEFT_DOWN (7): 向左优先，边缘向下
        [7, 4, 3, 0],  # 0°→LEFT_DOWN, 90°→UP_RIGHT, 180°→RIGHT_UP, 270°→DOWN_LEFT
    ]
    
    @classmethod
    def convert_visual_to_standard_direction(
        cls, 
        visual_direction: CropAddDirection, 
        orientation: int
    ) -> CropAddDirection:
        """
        Convert a visual (display-oriented) direction to standard coordinate system direction.
        
        Args:
            visual_direction: Direction as seen by user in the current display
            orientation: Image orientation in degrees (0, 90, 180, 270)
            
        Returns:
            Equivalent direction in standard coordinate system
            
        Example:
            # Image rotated 90° clockwise, user wants to go "down-right" visually
            # In standard coordinates, this means "left-up"
            result = convert_visual_to_standard_direction(
                CropAddDirection.DOWN_RIGHT, 90
            )
            # result == CropAddDirection.LEFT_UP
        """
        # Normalize orientation to 0-270 range
        orientation = orientation % 360
        if orientation not in [0, 90, 180, 270]:
            raise ValueError(f"Orientation must be 0, 90, 180, or 270 degrees, got {orientation}")
        
        # Get direction indices
        try:
            visual_index = cls._DIRECTION_ORDER.index(visual_direction)
        except ValueError:
            raise ValueError(f"Unknown visual direction: {visual_direction}")
        
        orientation_quarter = (orientation // 90) % 4
        
        # Look up mapped direction
        standard_index = cls._DIRECTION_MAPPING[visual_index][orientation_quarter]
        standard_direction = cls._DIRECTION_ORDER[standard_index]
        
        return standard_direction
    
    @classmethod
    def get_all_mappings(cls) -> Dict[int, Dict[CropAddDirection, CropAddDirection]]:
        """
        Get all direction mappings for debugging and testing purposes.
        
        Returns:
            Dictionary mapping orientation -> {visual_direction: standard_direction}
        """
        mappings = {}
        
        for orientation in [0, 90, 180, 270]:
            orientation_mappings = {}
            for visual_direction in cls._DIRECTION_ORDER:
                standard_direction = cls.convert_visual_to_standard_direction(
                    visual_direction, orientation
                )
                orientation_mappings[visual_direction] = standard_direction
            mappings[orientation] = orientation_mappings
            
        return mappings
    
    @classmethod
    def validate_mapping_consistency(cls) -> bool:
        """
        Validate that the direction mapping is consistent and bijective.
        
        Returns:
            True if mapping is valid, False otherwise
        """
        try:
            for orientation in [0, 90, 180, 270]:
                orientation_quarter = (orientation // 90) % 4
                
                # Check that all directions map to valid directions
                mapped_directions = []
                for i, visual_direction in enumerate(cls._DIRECTION_ORDER):
                    standard_index = cls._DIRECTION_MAPPING[i][orientation_quarter]
                    if not (0 <= standard_index < len(cls._DIRECTION_ORDER)):
                        return False
                    mapped_directions.append(standard_index)
                
                # Check that mapping is bijective (no two visual directions map to same standard)
                if len(set(mapped_directions)) != len(mapped_directions):
                    return False
                    
            return True
        except (IndexError, ValueError):
            return False


# Convenience function for direct usage
def convert_visual_to_standard_direction(
    visual_direction: CropAddDirection, 
    orientation: int
) -> CropAddDirection:
    """
    Convenience function to convert visual direction to standard direction.
    See OrientationDirectionMapper.convert_visual_to_standard_direction for details.
    """
    return OrientationDirectionMapper.convert_visual_to_standard_direction(
        visual_direction, orientation
    )