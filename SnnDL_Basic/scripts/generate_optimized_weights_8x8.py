#!/usr/bin/env python3

import os
import struct
import random

# 8x8网络架构配置
MESH_SIZE = 8
NUM_CORES_PER_PE = 4
NEURONS_PER_CORE = 4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 16
TOTAL_NODES = MESH_SIZE * MESH_SIZE  # 64

# 网络分层定义 (5层架构)
INPUT_LAYER = list(range(0, 8))      # PE 0-7: 输入层 (8个PE)
HIDDEN_LAYER_1 = list(range(8, 24))   # PE 8-23: 隐藏层1 (16个PE)
HIDDEN_LAYER_2 = list(range(24, 40))  # PE 24-39: 隐藏层2 (16个PE)
HIDDEN_LAYER_3 = list(range(40, 56))  # PE 40-55: 隐藏层3 (16个PE)
OUTPUT_LAYER = list(range(56, 64))   # PE 56-63: 输出层 (8个PE)

# 创建权重目录
weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights_8x8")
os.makedirs(weights_dir, exist_ok=True)

def generate_optimized_layer_weights_8x8(source_layer, target_layer, weight_strength=1.0, connection_type="full"):
    """生成8x8网络优化的层间连接权重"""
    weights = {}
    
    for target_pe in target_layer:
        # 为每个目标PE创建权重矩阵 (64*16 = 1024个神经元)
        weight_matrix = [0.0] * (NEURONS_PER_PE * (TOTAL_NODES * NEURONS_PER_PE))
        
        if connection_type == "selective":
            # 选择性连接：每个目标PE主要连接到2-3个特定源PE
            num_primary_sources = min(2, len(source_layer))  # 主要连接数
            target_offset = target_pe - min(target_layer)
            
            # 计算主要连接的源PE
            primary_sources = []
            for i in range(num_primary_sources):
                src_idx = (target_offset * num_primary_sources + i) % len(source_layer)
                primary_sources.append(source_layer[src_idx])
            
            # 次要连接的源PE
            secondary_sources = [pe for pe in source_layer if pe not in primary_sources]
            
            # 主要连接（强权重）
            for primary_source in primary_sources:
                connection_ratio = 0.8  # 80%神经元连接
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        if random.random() < connection_ratio:
                            source_neuron_global = primary_source * NEURONS_PER_PE + source_neuron_local
                            weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                            random.seed(target_pe * 10000 + primary_source * 1000 + target_neuron_local * 100 + source_neuron_local)
                            weight_value = weight_strength * (1.5 + 0.8 * random.random())  # 1.5-2.3倍强连接
                            weight_matrix[weight_index] = weight_value
            
            # 次要连接（中等权重）
            for secondary_source in secondary_sources:
                connection_ratio = 0.4  # 40%神经元连接
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        if random.random() < connection_ratio:
                            source_neuron_global = secondary_source * NEURONS_PER_PE + source_neuron_local
                            weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                            random.seed(target_pe * 20000 + secondary_source * 2000 + target_neuron_local * 200 + source_neuron_local)
                            weight_value = weight_strength * 0.5 * (0.8 + 0.6 * random.random())  # 0.4-0.7倍弱连接
                            weight_matrix[weight_index] = weight_value
                            
        elif connection_type == "competitive":
            # 竞争性连接：8个输出PE对应8个类别
            class_preference = target_pe - min(target_layer)  # PE56->0, PE57->1, ..., PE63->7
            
            for source_pe in source_layer:
                # 计算源PE的类别倾向
                source_class = source_pe % len(OUTPUT_LAYER)  # 映射到8个类别
                
                # 同类别强连接，异类别弱连接
                if source_class == class_preference:
                    strength_multiplier = 2.5  # 同类别强化
                    connection_ratio = 0.8
                else:
                    strength_multiplier = 0.15  # 异类别抑制
                    connection_ratio = 0.3
                
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        if random.random() < connection_ratio:
                            source_neuron_global = source_pe * NEURONS_PER_PE + source_neuron_local
                            weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                            random.seed(target_pe * 30000 + source_pe * 3000 + target_neuron_local * 300 + source_neuron_local)
                            weight_value = weight_strength * strength_multiplier * (0.9 + 0.2 * random.random())
                            weight_matrix[weight_index] = weight_value
                            
        elif connection_type == "sparse_full":
            # 稀疏全连接：降低连接密度以适应大规模网络
            connection_ratio = 0.3  # 只连接30%的神经元对
            for source_pe in source_layer:
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        if random.random() < connection_ratio:
                            source_neuron_global = source_pe * NEURONS_PER_PE + source_neuron_local
                            weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                            random.seed(target_pe * 10000 + source_pe * 1000 + target_neuron_local * 100 + source_neuron_local)
                            weight_value = weight_strength * (1.0 + 0.3 * random.random())  # 1.0-1.3倍连接
                            weight_matrix[weight_index] = weight_value
                            
        else:  # "full" - 全连接
            for source_pe in source_layer:
                for target_neuron_local in range(NEURONS_PER_PE):
                    for source_neuron_local in range(NEURONS_PER_PE):
                        source_neuron_global = source_pe * NEURONS_PER_PE + source_neuron_local
                        weight_index = target_neuron_local * (TOTAL_NODES * NEURONS_PER_PE) + source_neuron_global
                        random.seed(target_pe * 10000 + source_pe * 1000 + target_neuron_local * 100 + source_neuron_local)
                        weight_value = weight_strength * (1.0 + 0.2 * random.random())
                        weight_matrix[weight_index] = weight_value
        
        weights[target_pe] = weight_matrix
    
    return weights

print("🔗 生成8x8分层网络的优化权重文件...")
print(f"📊 网络规模: {TOTAL_NODES}个PE，总计{TOTAL_NODES * NEURONS_PER_PE}个神经元")

# 生成各层间的权重
layer_weights = {}

# 输入层 -> 隐藏层1 (选择性连接)
print("  🔄 生成输入层(PE 0-7) -> 隐藏层1(PE 8-23)权重...")
input_to_hidden1 = generate_optimized_layer_weights_8x8(INPUT_LAYER, HIDDEN_LAYER_1, 
                                                        weight_strength=12.0, connection_type="selective")
layer_weights.update(input_to_hidden1)
print(f"    输入层(8个PE) -> 隐藏层1(16个PE): 权重强度 12.0 (选择性连接)")

# 隐藏层1 -> 隐藏层2 (稀疏全连接)
print("  🔄 生成隐藏层1(PE 8-23) -> 隐藏层2(PE 24-39)权重...")
hidden1_to_hidden2 = generate_optimized_layer_weights_8x8(HIDDEN_LAYER_1, HIDDEN_LAYER_2, 
                                                          weight_strength=14.0, connection_type="sparse_full")
layer_weights.update(hidden1_to_hidden2)
print(f"    隐藏层1(16个PE) -> 隐藏层2(16个PE): 权重强度 14.0 (稀疏全连接)")

# 隐藏层2 -> 隐藏层3 (选择性连接)
print("  🔄 生成隐藏层2(PE 24-39) -> 隐藏层3(PE 40-55)权重...")
hidden2_to_hidden3 = generate_optimized_layer_weights_8x8(HIDDEN_LAYER_2, HIDDEN_LAYER_3, 
                                                          weight_strength=16.0, connection_type="selective")
layer_weights.update(hidden2_to_hidden3)
print(f"    隐藏层2(16个PE) -> 隐藏层3(16个PE): 权重强度 16.0 (选择性连接)")

# 隐藏层3 -> 输出层 (竞争性连接)
print("  🔄 生成隐藏层3(PE 40-55) -> 输出层(PE 56-63)权重...")
hidden3_to_output = generate_optimized_layer_weights_8x8(HIDDEN_LAYER_3, OUTPUT_LAYER, 
                                                         weight_strength=18.0, connection_type="competitive")
layer_weights.update(hidden3_to_output)
print(f"    隐藏层3(16个PE) -> 输出层(8个PE): 权重强度 18.0 (竞争性连接)")

print(f"\n💾 写入权重文件到 {weights_dir}...")

# 写入权重文件
files_written = 0
total_weights = 0
non_zero_weights = 0

for pe_id in range(TOTAL_NODES):
    filename = os.path.join(weights_dir, f"classification_weights_pe_{pe_id}.bin")
    
    if pe_id in layer_weights:
        weight_matrix = layer_weights[pe_id]
    else:
        # 没有特殊权重的PE使用全零权重
        weight_matrix = [0.0] * (NEURONS_PER_PE * (TOTAL_NODES * NEURONS_PER_PE))
    
    # 统计权重信息
    total_weights += len(weight_matrix)
    non_zero_weights += sum(1 for w in weight_matrix if w != 0.0)
    
    # 写入二进制文件
    with open(filename, 'wb') as f:
        for weight in weight_matrix:
            f.write(struct.pack('f', weight))
    
    files_written += 1
    
    # 显示进度
    if files_written % 16 == 0:
        progress = files_written / TOTAL_NODES * 100
        print(f"    进度: {progress:.0f}% ({files_written}/{TOTAL_NODES}个文件)")

print(f"\n✅ 权重文件生成完成！")
print(f"📊 统计信息:")
print(f"  总文件数: {files_written}")
print(f"  总权重数: {total_weights:,}")
print(f"  非零权重: {non_zero_weights:,}")
print(f"  稀疏率: {(1 - non_zero_weights/total_weights)*100:.1f}%")

# 计算各层的权重统计
print(f"\n📈 分层权重统计:")

def analyze_layer_weights(layer_name, layer_pes, weights_dict):
    layer_total = 0
    layer_nonzero = 0
    
    for pe_id in layer_pes:
        if pe_id in weights_dict:
            weight_matrix = weights_dict[pe_id]
            layer_total += len(weight_matrix)
            layer_nonzero += sum(1 for w in weight_matrix if w != 0.0)
    
    if layer_total > 0:
        sparsity = (1 - layer_nonzero/layer_total) * 100
        avg_weight = sum(abs(w) for pe_id in layer_pes if pe_id in weights_dict 
                        for w in weights_dict[pe_id] if w != 0.0) / max(layer_nonzero, 1)
        print(f"  {layer_name}: {layer_nonzero:,}非零权重/{layer_total:,}总权重 (稀疏率{sparsity:.1f}%, 平均权重{avg_weight:.3f})")

analyze_layer_weights("隐藏层1", HIDDEN_LAYER_1, layer_weights)
analyze_layer_weights("隐藏层2", HIDDEN_LAYER_2, layer_weights)
analyze_layer_weights("隐藏层3", HIDDEN_LAYER_3, layer_weights)
analyze_layer_weights("输出层", OUTPUT_LAYER, layer_weights)

print(f"\n🎯 8x8网络权重生成策略:")
print(f"  • 输入层(8) -> 隐藏层1(16): 选择性连接，权重强度12.0")
print(f"  • 隐藏层1(16) -> 隐藏层2(16): 稀疏全连接，权重强度14.0")
print(f"  • 隐藏层2(16) -> 隐藏层3(16): 选择性连接，权重强度16.0")
print(f"  • 隐藏层3(16) -> 输出层(8): 竞争性连接，权重强度18.0")
print(f"\n🚀 权重文件已准备完成，可用于8x8网络分类任务！")