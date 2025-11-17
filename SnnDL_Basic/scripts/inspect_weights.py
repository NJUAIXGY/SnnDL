#!/usr/bin/env python3

import struct
import os
import numpy as np

# 配置参数
NEURONS_PER_PE = 16
TOTAL_NODES = 16
test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

# 网络分层定义
INPUT_LAYER = list(range(0, 4))      # PE 0-3: 输入层
HIDDEN_LAYER_1 = list(range(4, 8))   # PE 4-7: 隐藏层1
HIDDEN_LAYER_2 = list(range(8, 12))  # PE 8-11: 隐藏层2
OUTPUT_LAYER = list(range(12, 16))   # PE 12-15: 输出层

def read_weight_file(pe_id):
    """读取并分析权重文件"""
    weight_file = os.path.join(test_dir, f"classification_weights_pe_{pe_id}.bin")
    
    if not os.path.exists(weight_file):
        print(f"❌ 权重文件不存在: {weight_file}")
        return None
    
    with open(weight_file, 'rb') as f:
        data = f.read()
    
    # 每个float是4字节
    expected_size = NEURONS_PER_PE * TOTAL_NODES * NEURONS_PER_PE * 4
    actual_size = len(data)
    
    if actual_size != expected_size:
        print(f"⚠️ PE{pe_id} 文件大小异常: {actual_size} 字节 (期望: {expected_size} 字节)")
        return None
    
    # 解析float数据
    weights = []
    for i in range(0, len(data), 4):
        weight = struct.unpack('f', data[i:i+4])[0]
        weights.append(weight)
    
    return weights

def analyze_weights(pe_id, weights):
    """分析权重分布"""
    weights = np.array(weights)
    
    # 基本统计
    non_zero_count = np.sum(np.abs(weights) > 0.001)
    zero_count = len(weights) - non_zero_count
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    mean_weight = np.mean(np.abs(weights))
    
    # 确定层类型
    layer_name = "输入层" if pe_id in INPUT_LAYER else \
                 "隐藏层1" if pe_id in HIDDEN_LAYER_1 else \
                 "隐藏层2" if pe_id in HIDDEN_LAYER_2 else "输出层"
    
    print(f"\n📊 PE{pe_id} ({layer_name}) 权重分析:")
    print(f"  总权重数: {len(weights)}")
    print(f"  非零权重: {non_zero_count} ({non_zero_count/len(weights)*100:.1f}%)")
    print(f"  零权重: {zero_count} ({zero_count/len(weights)*100:.1f}%)")
    print(f"  最大权重: {max_weight:.3f}")
    print(f"  最小权重: {min_weight:.3f}")
    print(f"  平均权重: {mean_weight:.3f}")
    
    # 权重分布
    weight_ranges = [
        (0.0, 0.001, "接近零"),
        (0.001, 1.0, "小权重"),
        (1.0, 10.0, "中权重"),
        (10.0, 20.0, "大权重"),
        (20.0, float('inf'), "超大权重")
    ]
    
    print(f"  权重分布:")
    for min_val, max_val, desc in weight_ranges:
        if max_val == float('inf'):
            count = np.sum(weights >= min_val)
        else:
            count = np.sum((weights >= min_val) & (weights < max_val))
        if count > 0:
            print(f"    {desc}: {count} 个 ({count/len(weights)*100:.1f}%)")
    
    return {
        'pe_id': pe_id,
        'layer': layer_name,
        'total': len(weights),
        'non_zero': non_zero_count,
        'max': max_weight,
        'min': min_weight,
        'mean': mean_weight
    }

def check_connectivity(pe_id, weights):
    """检查连接性"""
    weights = np.array(weights)
    
    # 重新组织为矩阵形式: [目标神经元][源神经元]
    weight_matrix = weights.reshape(NEURONS_PER_PE, TOTAL_NODES * NEURONS_PER_PE)
    
    print(f"\n🔗 PE{pe_id} 连接性分析:")
    
    # 检查与其他PE的连接
    for src_pe in range(TOTAL_NODES):
        start_neuron = src_pe * NEURONS_PER_PE
        end_neuron = (src_pe + 1) * NEURONS_PER_PE
        
        # 提取与源PE的所有连接权重
        connections = weight_matrix[:, start_neuron:end_neuron]
        non_zero_connections = np.sum(np.abs(connections) > 0.001)
        
        if non_zero_connections > 0:
            max_conn = np.max(connections)
            avg_conn = np.mean(connections[np.abs(connections) > 0.001])
            
            src_layer = "输入层" if src_pe in INPUT_LAYER else \
                        "隐藏层1" if src_pe in HIDDEN_LAYER_1 else \
                        "隐藏层2" if src_pe in HIDDEN_LAYER_2 else "输出层"
            
            print(f"  从PE{src_pe}({src_layer}): {non_zero_connections}个连接, 最大={max_conn:.3f}, 平均={avg_conn:.3f}")

print("🔍 权重文件检查报告")
print("="*50)

# 检查所有权重文件
all_stats = []
for pe_id in range(TOTAL_NODES):
    weights = read_weight_file(pe_id)
    if weights is not None:
        stats = analyze_weights(pe_id, weights)
        all_stats.append(stats)
        check_connectivity(pe_id, weights)

print(f"\n📋 权重文件总结:")
print(f"✅ 成功加载 {len(all_stats)} 个权重文件")

# 按层统计
for layer_nodes, layer_name in [
    (INPUT_LAYER, "输入层"),
    (HIDDEN_LAYER_1, "隐藏层1"), 
    (HIDDEN_LAYER_2, "隐藏层2"),
    (OUTPUT_LAYER, "输出层")
]:
    layer_stats = [s for s in all_stats if s['pe_id'] in layer_nodes]
    if layer_stats:
        avg_non_zero = np.mean([s['non_zero'] for s in layer_stats])
        avg_max = np.mean([s['max'] for s in layer_stats])
        print(f"  {layer_name}: 平均 {avg_non_zero:.0f} 个非零权重, 平均最大权重 {avg_max:.3f}")