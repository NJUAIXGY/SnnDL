#!/usr/bin/env python3

import os

# === 8x8网络配置 ===
MESH_SIZE = 8
NUM_CORES_PER_PE = 4
NEURONS_PER_CORE = 4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 16
INPUT_LAYER = list(range(0, 8))  # PE 0-7

# 8类分类任务配置 (扩展频率范围)
CLASS_A_FREQ = 40   # 类别A: 低频规律脉冲 (40Hz)
CLASS_B_FREQ = 60   # 类别B: 低中频突发脉冲 (60Hz)
CLASS_C_FREQ = 80   # 类别C: 中频突发脉冲 (80Hz)
CLASS_D_FREQ = 100  # 类别D: 中高频混合模式 (100Hz)
CLASS_E_FREQ = 120  # 类别E: 高频混合模式 (120Hz)
CLASS_F_FREQ = 150  # 类别F: 高频稀疏脉冲 (150Hz)
CLASS_G_FREQ = 180  # 类别G: 超高频模式 (180Hz)
CLASS_H_FREQ = 200  # 类别H: 极高频稀疏脉冲 (200Hz)

def create_complex_spike_data_8x8(filename, pe_id, class_type, duration_us=200.0):
    """创建8类复杂分类任务的脉冲数据"""
    spikes = []
    import random
    random.seed(42 + pe_id + class_type * 100)  # 确保可重复性但区分类别
    
    start_time_us = 5.0  # 延迟启动
    end_time = start_time_us + duration_us
    
    frequencies = [CLASS_A_FREQ, CLASS_B_FREQ, CLASS_C_FREQ, CLASS_D_FREQ, 
                   CLASS_E_FREQ, CLASS_F_FREQ, CLASS_G_FREQ, CLASS_H_FREQ]
    
    frequency = frequencies[class_type]
    interval_us = 1000000.0 / frequency
    
    if class_type == 0:  # 类别A: 低频规律脉冲 + 轻微噪声
        current_time = start_time_us
        while current_time < end_time:
            # 规律脉冲 + 时间抖动
            jitter = random.uniform(-0.15, 0.15) * interval_us
            current_time += interval_us + jitter
            if current_time < end_time:
                spikes.append(current_time)
                
            # 偶尔添加额外噪声脉冲
            if random.random() < 0.08:  # 8%概率
                noise_spike = current_time + random.uniform(1.0, 4.0)
                if noise_spike < end_time:
                    spikes.append(noise_spike)
                    
    elif class_type == 1:  # 类别B: 低中频突发脉冲
        burst_duration = 15.0
        quiet_duration = 25.0
        current_time = start_time_us
        
        while current_time < end_time:
            # 突发期间
            burst_end = current_time + burst_duration
            while current_time < burst_end and current_time < end_time:
                current_time += interval_us + random.uniform(-0.1, 0.1) * interval_us
                if current_time < burst_end and current_time < end_time:
                    spikes.append(current_time)
            current_time += quiet_duration
            
    elif class_type == 2:  # 类别C: 中频突发脉冲 (原有逻辑)
        burst_duration = 12.0
        quiet_duration = 20.0
        current_time = start_time_us
        
        while current_time < end_time:
            burst_end = current_time + burst_duration
            while current_time < burst_end and current_time < end_time:
                current_time += interval_us + random.uniform(-0.1, 0.1) * interval_us
                if current_time < burst_end and current_time < end_time:
                    spikes.append(current_time)
            current_time += quiet_duration
            
    elif class_type == 3:  # 类别D: 中高频混合模式
        current_time = start_time_us
        
        # 混合模式：规律 + 随机间歇
        while current_time < end_time:
            # 基础规律脉冲
            current_time += interval_us * random.uniform(0.8, 1.2)
            if current_time < end_time:
                spikes.append(current_time)
                
            # 随机决定是否添加间歇
            if random.random() < 0.25:  # 25%概率
                current_time += interval_us * random.uniform(1.5, 3.0)
                
    elif class_type == 4:  # 类别E: 高频混合模式 (原有逻辑增强)
        current_time = start_time_us
        
        while current_time < end_time:
            # 基础规律脉冲
            current_time += interval_us * random.uniform(0.7, 1.3)
            if current_time < end_time:
                spikes.append(current_time)
                
            # 随机决定是否添加短突发
            if random.random() < 0.3:  # 30%概率
                for i in range(random.randint(1, 3)):
                    burst_spike = current_time + i * 2.0 + random.uniform(0, 2.0)
                    if burst_spike < end_time:
                        spikes.append(burst_spike)
                        
    elif class_type == 5:  # 类别F: 高频稀疏脉冲
        current_time = start_time_us
        
        # 稀疏高频模式：短时间高密度，然后长间隔
        while current_time < end_time:
            # 高频密集期
            dense_duration = 8.0
            dense_end = current_time + dense_duration
            while current_time < dense_end and current_time < end_time:
                current_time += interval_us * random.uniform(0.5, 0.8)
                if current_time < dense_end and current_time < end_time:
                    spikes.append(current_time)
            
            # 稀疏间隔期
            current_time += random.uniform(20.0, 35.0)
            
    elif class_type == 6:  # 类别G: 超高频模式
        current_time = start_time_us
        
        # 三重突发模式：短突发 + 中等间隔 + 再突发
        while current_time < end_time:
            # 第一轮突发
            for i in range(random.randint(2, 4)):
                current_time += interval_us * random.uniform(0.6, 0.9)
                if current_time < end_time:
                    spikes.append(current_time)
            
            # 短间隔
            current_time += interval_us * random.uniform(2.0, 4.0)
            
            # 第二轮突发
            for i in range(random.randint(1, 3)):
                current_time += interval_us * random.uniform(0.7, 1.0)
                if current_time < end_time:
                    spikes.append(current_time)
            
            # 长间隔
            current_time += random.uniform(15.0, 25.0)
            
    elif class_type == 7:  # 类别H: 极高频稀疏脉冲 (原有逻辑)
        current_time = start_time_us
        
        # 极稀疏超高频模式
        while current_time < end_time:
            # 极短高频突发
            burst_duration = 8.0
            burst_end = current_time + burst_duration
            while current_time < burst_end and current_time < end_time:
                current_time += interval_us * random.uniform(0.4, 0.7)
                if current_time < burst_end and current_time < end_time:
                    spikes.append(current_time)
            
            # 长静默期
            current_time += random.uniform(32.0, 45.0)
    
    # 确保每个神经元至少有2个脉冲
    min_spikes_per_neuron = 2
    total_min_spikes = NEURONS_PER_PE * min_spikes_per_neuron
    
    while len(spikes) < total_min_spikes:
        # 在时间范围内添加随机脉冲
        extra_spike = start_time_us + random.uniform(0, duration_us)
        spikes.append(extra_spike)
    
    # 排序并去重
    spikes = sorted(list(set(spikes)))
    
    # 写入文件
    with open(filename, 'w') as f:
        f.write(f"# 类别{chr(65 + class_type)}脉冲数据 (PE {pe_id}) - 频率: {frequency}Hz\n")
        f.write(f"# 总脉冲数: {len(spikes)}\n")
        f.write(f"# 持续时间: {duration_us}us\n")
        
        # 为每个脉冲分配神经元
        for i, spike_time in enumerate(spikes):
            local_neuron = i % NEURONS_PER_PE  # 循环分配到16个神经元（本PE内）
            global_neuron = pe_id * NEURONS_PER_PE + local_neuron
            ts_us = int(round(spike_time))  # SpikeSource TEXT格式: neuron_id timestamp（整数）
            f.write(f"{global_neuron} {ts_us}\n")
    
    return len(spikes)

# 创建脉冲数据目录
spike_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spike_data_8x8")
os.makedirs(spike_dir, exist_ok=True)

print("🧠 生成8x8网络的8类脉冲数据...")
print("=" * 50)

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
frequencies = [CLASS_A_FREQ, CLASS_B_FREQ, CLASS_C_FREQ, CLASS_D_FREQ, 
               CLASS_E_FREQ, CLASS_F_FREQ, CLASS_G_FREQ, CLASS_H_FREQ]

total_spikes = 0

for pe_id in INPUT_LAYER:
    class_type = pe_id  # PE 0->类别A, PE 1->类别B, ..., PE 7->类别H
    class_name = class_names[class_type]
    frequency = frequencies[class_type]
    
    filename = os.path.join(spike_dir, f"complex_input_pe_{pe_id}_class_{class_name}.txt")
    
    print(f"🔄 生成PE{pe_id} - 类别{class_name} ({frequency}Hz)...")
    spike_count = create_complex_spike_data_8x8(filename, pe_id, class_type, duration_us=200.0)
    total_spikes += spike_count
    
    print(f"  ✅ 完成: {spike_count}个脉冲事件")

print(f"\n📊 8x8网络脉冲数据统计:")
print(f"  总文件数: {len(INPUT_LAYER)}")
print(f"  总脉冲数: {total_spikes}")
print(f"  平均每PE: {total_spikes // len(INPUT_LAYER)}个脉冲")
print(f"  数据目录: {spike_dir}")

print(f"\n🎯 8类脉冲模式特征:")
print(f"  类别A (40Hz): 规律脉冲 + 8%噪声")
print(f"  类别B (60Hz): 15ms突发 + 25ms静默")
print(f"  类别C (80Hz): 12ms突发 + 20ms静默")
print(f"  类别D (100Hz): 规律脉冲 + 随机间歇")
print(f"  类别E (120Hz): 规律脉冲 + 30%短突发")
print(f"  类别F (150Hz): 8ms高密度 + 20-35ms稀疏")
print(f"  类别G (180Hz): 三重突发模式")
print(f"  类别H (200Hz): 8ms极高频 + 32-45ms静默")

print(f"\n🚀 8x8网络脉冲数据生成完成！")
print(f"💡 这些数据可用于测试8x8分层网络的分类能力")