# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:09:21 2024

@author: SAM
"""

import os

# 替换为你需要处理的文件所在的目录路径
directory = "C:/Users/SAM/Documents/game theory and machine learning/codes for networks game/data"

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件名中是否包含 'W_1.0'
    if 'U_1' in filename:
        # 生成新的文件名，将 'W_1.0' 替换为 'W_1'
        new_filename = filename.replace('U_1', 'U_1.0')
        
        # 获取完整的文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

print("All files have been renamed.")
