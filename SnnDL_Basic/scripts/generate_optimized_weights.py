#!/usr/bin/env python3

import os
import struct
import random

# 网络架构配置
MESH_SIZE = 4
NUM_CORES_PER_PE = 4
NEURONS_PER_CORE = 4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 16
TOTAL_NODES = MESH_SIZE * MESH_SIZE  # 16

# 网络分层定义
INPUT_LAYER = list(range(0, 4))      # PE 0-3: 输入层
HIDDEN_LAYER_1 = list(range(4, 8))   # PE 4-7: 隐藏层1
HIDDEN_LAYER_2 = list(range(8, 12))  # PE 8-11: 隐藏层2
OUTPUT_LAYER = list(range(12, 16))   # PE 12-15: 输出层

test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
os.makedirs(test_dir, exist_ok=True)

def generate_optimized_layer_weights(source_layer, target_layer, weight_strength=1.0, connection_type="full"):
    """生成优化的层间连接权重"""
    weights = {}
    
    for target_pe in target_layer:
        # 为每个目标PE创建权重矩阵
        weight_matrix = [0.0] * (NEURONS_PER_PE * (TOTAL_NODES * NEURONS_PER_PE))
        
        if connection_type == "selective":
            # 选择性连接：每个目标PE主要连接到特定源PE
            primary_source = source_layer[target_pe % len(source_layer)]
            secondary_sources = [pe for pe in source_layer if pe != primary_source]
            
            # 主要连接（强权重）
            for target_neuron_local in range(NEURONS_PER_PE):
                for source_neuron_local in range(NEURONS_PER_PE):
                    source_neuron_global = primary_source * NEURONS_PER_PE + source_neuron_local
                    weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                    random.seed(target_pe * 1000 + primary_source * 100 + target_neuron_local * 10 + source_neuron_local)
                    weight_value = weight_strength * (1.2 + 0.6 * random.random())  # 1.2-1.8倍强连接
                    weight_matrix[weight_index] = weight_value
            
            # 次要连接（中等权重）
            for secondary_source in secondary_sources:
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        source_neuron_global = secondary_source * NEURONS_PER_PE + source_neuron_local
                        weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                        random.seed(target_pe * 2000 + secondary_source * 200 + target_neuron_local * 20 + source_neuron_local)
                        weight_value = weight_strength * 0.6 * (0.8 + 0.4 * random.random())  # 0.48-0.72倍弱连接
                        weight_matrix[weight_index] = weight_value
                        
        elif connection_type == "competitive":
            # 竞争性连接：每个输出PE偏向一个类别
            class_preference = target_pe - min(target_layer)  # PE12->0, PE13->1, PE14->2, PE15->3
            
            for source_pe in source_layer:
                source_class = source_pe % len(source_layer)  # 映射到类别
                
                # 同类别强连接，异类别弱连接
                if source_class == class_preference:
                    strength_multiplier = 2.0  # 同类别强化
                else:
                    strength_multiplier = 0.2   # 异类别抑制
                
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        source_neuron_global = source_pe * NEURONS_PER_PE + source_neuron_local
                        weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                        random.seed(target_pe * 3000 + source_pe * 300 + target_neuron_local * 30 + source_neuron_local)
                        weight_value = weight_strength * strength_multiplier * (0.9 + 0.2 * random.random())
                        weight_matrix[weight_index] = weight_value
                        
        else:  # "full" - 全连接
            for source_pe in source_layer:
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        source_neuron_global = source_pe * NEURONS_PER_PE + source_neuron_local
                        weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                        random.seed(target_pe * 1000 + source_pe * 100 + target_neuron_local * 10 + source_neuron_local)
                        weight_value = weight_strength * (1.0 + 0.2 * random.random())  # 1.0-1.2倍连接
                        weight_matrix[weight_index] = weight_value
        
        weights[target_pe] = weight_matrix
    
    return weights

print("🔗 生成优化的分层权重文件...")

# 生成各层间的权重
layer_weights = {}

# 输入层 -> 隐藏层1 (选择性连接，增强激活)
input_to_hidden1 = generate_optimized_layer_weights(INPUT_LAYER, HIDDEN_LAYER_1, 
                                                    weight_strength=12.0, connection_type="selective")
layer_weights.update(input_to_hidden1)
print(f"  输入层(PE 0-3) -> 隐藏层1(PE 4-7): 权重强度 10.0 (选择性连接)")

# 隐藏层1 -> 隐藏层2 (选择性连接，强化传播)
hidden1_to_hidden2 = generate_optimized_layer_weights(HIDDEN_LAYER_1, HIDDEN_LAYER_2, 
                                                      weight_strength=14.0, connection_type="selective")
layer_weights.update(hidden1_to_hidden2)
print(f"  隐藏层1(PE 4-7) -> 隐藏层2(PE 8-11): 权重强度 12.0 (选择性连接)")

# 隐藏层2 -> 输出层 (竞争性连接，增强分类区分度)
hidden2_to_output = generate_optimized_layer_weights(HIDDEN_LAYER_2, OUTPUT_LAYER, 
                                                     weight_strength=18.0, connection_type="competitive")
layer_weights.update(hidden2_to_output)
print(f"  隐藏层2(PE 8-11) -> 输出层(PE 12-15): 权重强度 15.0 (竞争性连接)")

# 为输入层设置增强的自连接权重
for pe_id in INPUT_LAYER:
    if pe_id not in layer_weights:
        weight_matrix = [0.0] * (NEURONS_PER_PE * (TOTAL_NODES * NEURONS_PER_PE))
        # 设置自连接权重增强输入层激活
        for neuron_local in range(NEURONS_PER_PE):
            for self_neuron in range(NEURONS_PER_PE):
                source_neuron_global = pe_id * NEURONS_PER_PE + self_neuron
                weight_index = neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                if neuron_local == self_neuron:
                    weight_matrix[weight_index] = 0.5  # 自激励
                else:
                    weight_matrix[weight_index] = 0.2  # 同PE内互连
        layer_weights[pe_id] = weight_matrix

# 写入权重文件
for pe_id in range(TOTAL_NODES):
    weight_file = os.path.join(test_dir, f"classification_weights_pe_{pe_id}.bin")
    
    if pe_id in layer_weights:
        weights = layer_weights[pe_id]
    else:
        # 默认权重：较小的随机值
        random.seed(pe_id * 1000)
        weights = [0.05 + 0.05 * random.random() for _ in range(NEURONS_PER_PE * (TOTAL_NODES * NEURONS_PER_PE))]
    
    with open(weight_file, 'wb') as f:
        for w in weights:
            f.write(struct.pack('f', w))
    
    layer_name = "输入层" if pe_id in INPUT_LAYER else \
                 "隐藏层1" if pe_id in HIDDEN_LAYER_1 else \
                 "隐藏层2" if pe_id in HIDDEN_LAYER_2 else "输出层"
    
    # 统计权重信息
    non_zero_weights = sum(1 for w in weights if abs(w) > 0.01)
    max_weight = max(weights)
    avg_weight = sum(abs(w) for w in weights) / len(weights)
    
    print(f"  PE{pe_id} ({layer_name}): {non_zero_weights}个有效权重, 最大={max_weight:.3f}, 平均={avg_weight:.3f}")

print(f"\n✅ 优化权重文件生成完成！")
print(f"📁 文件位置: {test_dir}/classification_weights_pe_{{0-15}}.bin")
