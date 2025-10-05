#!/usr/bin/env python3
"""
修复8x8分类网络脉冲数据文件格式
将格式从 'timestamp neuron_id' 转换为 'neuron_id timestamp'
"""

import os
import shutil
import sys

def fix_spike_data_format(input_file, output_file):
    """
    转换脉冲数据文件格式
    从 'timestamp neuron_id' 转换为 'neuron_id timestamp'
    """
    print(f"修复文件格式: {input_file} -> {output_file}")
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                outfile.write(line + '\n')
                continue
            
            try:
                # 解析原格式: timestamp neuron_id
                parts = line.split()
                if len(parts) != 2:
                    print(f"警告: 第{line_num}行格式不正确: {line}")
                    continue
                
                timestamp = float(parts[0])
                neuron_id = int(parts[1])
                
                # 写入新格式: neuron_id timestamp
                # 将时间戳转换为微秒整数
                timestamp_us = int(timestamp * 1000)  # 转换为微秒
                outfile.write(f"{neuron_id} {timestamp_us}\n")
                
            except (ValueError, IndexError) as e:
                print(f"错误: 第{line_num}行解析失败: {line} - {e}")
                continue
    
    print(f"文件格式修复完成: {output_file}")

def main():
    # 数据文件目录
    spike_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spike_data_8x8")
    
    if not os.path.exists(spike_data_dir):
        print(f"错误: 数据目录不存在: {spike_data_dir}")
        sys.exit(1)
    
    print(f"修复8x8脉冲数据文件格式...")
    print(f"数据目录: {spike_data_dir}")
    
    # 创建备份目录
    backup_dir = spike_data_dir + "_backup"
    if not os.path.exists(backup_dir):
        print(f"创建备份目录: {backup_dir}")
        shutil.copytree(spike_data_dir, backup_dir)
    else:
        print(f"备份目录已存在: {backup_dir}")
    
    # 处理所有数据文件
    files_processed = 0
    for filename in os.listdir(spike_data_dir):
        if filename.endswith('.txt') and filename.startswith('complex_input_pe_'):
            input_file = os.path.join(spike_data_dir, filename)
            temp_file = input_file + '.tmp'
            
            # 修复格式到临时文件
            fix_spike_data_format(input_file, temp_file)
            
            # 替换原文件
            shutil.move(temp_file, input_file)
            files_processed += 1
    
    print(f"\n✅ 完成! 共修复了{files_processed}个文件")
    print(f"原始文件已备份到: {backup_dir}")
    print(f"格式已从 'timestamp neuron_id' 转换为 'neuron_id timestamp'")

if __name__ == "__main__":
    main()