#!/usr/bin/env python3
"""
ColorChecker数据处理脚本
将ColorChecker24_After_Nov2014.txt中的Lab数据转换为XYZ格式，并保存为JSON文件
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 添加divere模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from divere.core.color_science import lab_d50_to_xyz_d50


def get_standard_patch_mapping() -> Dict[str, str]:
    """
    获取Nov2014.txt中色块ID到标准ColorChecker 24色块ID的映射
    
    Nov2014.txt使用A1-F4的标记方式（6行4列），按列优先排列
    Nov2014.txt的顺序：A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2,A3,B3,C3,D3,E3,F3,A4,B4,C4,D4,E4,F4
    标准顺序应该是：A1,A2,A3,A4,A5,A6,B1,B2,B3,B4,B5,B6,C1,C2,C3,C4,C5,C6,D1,D2,D3,D4,D5,D6
    
    Returns:
        Dict[str, str]: Nov2014.txt的色块ID到标准色块ID的映射
    """
    # Nov2014.txt中的色块按列优先顺序（6行4列）
    nov2014_order = [
        'A1', 'B1', 'C1', 'D1', 'E1', 'F1',  # 第1列
        'A2', 'B2', 'C2', 'D2', 'E2', 'F2',  # 第2列
        'A3', 'B3', 'C3', 'D3', 'E3', 'F3',  # 第3列
        'A4', 'B4', 'C4', 'D4', 'E4', 'F4'   # 第4列
    ]
    
    # 标准ColorChecker 24色块排列（4行6列，按行优先）
    standard_order = [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6',  # 第1行
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6',  # 第2行  
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6',  # 第3行
        'D1', 'D2', 'D3', 'D4', 'D5', 'D6'   # 第4行
    ]
    
    # 创建映射：Nov2014.txt的ID -> 标准ID
    mapping = {}
    for i, nov_id in enumerate(nov2014_order):
        mapping[nov_id] = standard_order[i]
    
    return mapping


def parse_colorchecker_txt(file_path: Path) -> Dict[str, Tuple[float, float, float]]:
    """
    解析ColorChecker24_After_Nov2014.txt文件
    将Nov2014.txt中的色块ID映射为标准ColorChecker 24色块ID
    
    Returns:
        Dict[str, Tuple[float, float, float]]: 标准色块ID到Lab值的映射
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取BEGIN_DATA和END_DATA之间的数据
    data_section = re.search(r'BEGIN_DATA\s*\n(.*?)\nEND_DATA', content, re.DOTALL)
    if not data_section:
        raise ValueError("无法找到数据部分")
    
    # 获取色块ID映射
    patch_mapping = get_standard_patch_mapping()
    
    lab_data = {}
    for line in data_section.group(1).strip().split('\n'):
        if line.strip():
            parts = line.split('\t')
            if len(parts) >= 4:
                nov2014_patch_id = parts[0].strip()
                # 将逗号替换为点号以处理欧式小数表示
                l_val = float(parts[1].replace(',', '.'))
                a_val = float(parts[2].replace(',', '.'))
                b_val = float(parts[3].replace(',', '.'))
                
                # 将Nov2014.txt的色块ID映射为标准色块ID
                if nov2014_patch_id in patch_mapping:
                    standard_patch_id = patch_mapping[nov2014_patch_id]
                    lab_data[standard_patch_id] = (l_val, a_val, b_val)
                else:
                    print(f"警告: 无法映射色块ID: {nov2014_patch_id}")
    
    return lab_data


def convert_lab_to_xyz(lab_data: Dict[str, Tuple[float, float, float]]) -> Dict[str, List[float]]:
    """
    将Lab数据转换为XYZ数据
    
    Args:
        lab_data: 色块ID到Lab值的映射
        
    Returns:
        Dict[str, List[float]]: 色块ID到XYZ值列表的映射
    """
    xyz_data = {}
    for patch_id, lab_values in lab_data.items():
        xyz_array = lab_d50_to_xyz_d50(lab_values)
        xyz_data[patch_id] = xyz_array.tolist()
    
    return xyz_data


def create_output_json(xyz_data: Dict[str, List[float]]) -> Dict:
    """
    创建输出JSON结构
    
    Args:
        xyz_data: 色块ID到XYZ值的映射
        
    Returns:
        Dict: 完整的JSON输出结构
    """
    return {
        "description": "ColorChecker 24色块原始测量值在XYZ色彩空间的数值",
        "type": "XYZ", 
        "color_space": "XYZ(D50)",
        "conversion_path": "Lab(D50) → XYZ(D50)",
        "white_point": "D50",
        "source": "ColorChecker24_After_Nov2014.txt (X-Rite official measurement)",
        "data": xyz_data
    }


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    input_file = script_dir / "ColorChecker24_After_Nov2014.txt"
    output_file = script_dir / "original_color_cc24data.json"
    
    try:
        # 解析输入文件
        print(f"正在读取: {input_file}")
        lab_data = parse_colorchecker_txt(input_file)
        print(f"成功解析 {len(lab_data)} 个色块的Lab数据")
        
        # 转换为XYZ
        print("正在转换Lab -> XYZ...")
        xyz_data = convert_lab_to_xyz(lab_data)
        
        # 创建输出JSON
        output_data = create_output_json(xyz_data)
        
        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"转换完成! 输出文件: {output_file}")
        print(f"共处理 {len(xyz_data)} 个色块")
        
        # 显示前几个转换结果作为验证
        print("\n前几个转换结果:")
        for i, (patch_id, xyz_values) in enumerate(list(xyz_data.items())[:3]):
            lab_values = lab_data[patch_id]
            print(f"{patch_id}: Lab{lab_values} -> XYZ{xyz_values}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())