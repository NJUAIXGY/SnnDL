#!/usr/bin/env python3

import os

# === 网络配置 ===
MESH_SIZE = 4
NUM_CORES_PER_PE = 4
NEURONS_PER_CORE = 4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 16
INPUT_LAYER = list(range(0, 4))

# 分类任务配置
CLASS_A_FREQ = 40   # 类别A: 低频规律脉冲 (40Hz)
CLASS_B_FREQ = 80   # 类别B: 中频突发脉冲 (80Hz)
CLASS_C_FREQ = 120  # 类别C: 高频混合模式 (120Hz)
CLASS_D_FREQ = 200  # 类别D: 超高频稀疏脉冲 (200Hz)

def create_complex_spike_data(filename, pe_id, class_type, duration_us=180.0):
    """创建4类复杂分类任务的脉冲数据"""
    spikes = []
    import random
    random.seed(42 + pe_id + class_type * 100)  # 确保可重复性但区分类别
    
    start_time_us = 5.0  # 延迟启动
    end_time = start_time_us + duration_us
    
    if class_type == 0:  # 类别A: 低频规律脉冲 + 轻微噪声
        frequency = CLASS_A_FREQ
        interval_us = 1000000.0 / frequency
        current_time = start_time_us
        
        while current_time < end_time:
            # 规律脉冲 + 时间抖动
            jitter = random.uniform(-0.15, 0.15) * interval_us
            current_time += interval_us + jitter
            if current_time < end_time:
                spikes.append(current_time)
                
            # 偶尔添加额外噪声脉冲
            if random.random() < 0.1:  # 10%概率
                noise_spike = current_time + random.uniform(1.0, 5.0)
                if noise_spike < end_time:
                    spikes.append(noise_spike)
                    
    elif class_type == 1:  # 类别B: 中频突发脉冲
        frequency = CLASS_B_FREQ
        interval_us = 1000000.0 / frequency
        
        # 突发模式：短时间密集，然后间隔
        burst_duration = 12.0
        quiet_duration = 20.0
        current_time = start_time_us
        
        while current_time < end_time:
            # 突发期间
            burst_end = current_time + burst_duration
            while current_time < burst_end and current_time < end_time:
                current_time += interval_us + random.uniform(-0.1, 0.1) * interval_us
                if current_time < burst_end and current_time < end_time:
                    spikes.append(current_time)
            current_time += quiet_duration
            
    elif class_type == 2:  # 类别C: 高频混合模式
        frequency = CLASS_C_FREQ
        interval_us = 1000000.0 / frequency
        current_time = start_time_us
        
        # 混合模式：规律 + 随机 + 短突发
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
                        
    else:  # 类别D: 超高频稀疏脉冲
        frequency = CLASS_D_FREQ
        interval_us = 1000000.0 / frequency
        current_time = start_time_us
        
        # 稀疏高频：短时间内极高频率，然后长间隔
        active_duration = 8.0  # 8ms活跃期
        silent_duration = 32.0  # 32ms安静期
        
        while current_time < end_time:
            # 高频活跃期
            active_end = current_time + active_duration
            while current_time < active_end and current_time < end_time:
                current_time += interval_us + random.uniform(-0.02, 0.02) * interval_us
                if current_time < active_end and current_time < end_time:
                    spikes.append(current_time)
            current_time += silent_duration
    
    # 确保有足够的脉冲 - 增加最小脉冲数量
    min_spikes_needed = 20  # 显著增加最小脉冲数
    if len(spikes) < min_spikes_needed:
        # 基于频率计算应有的脉冲数
        frequency = [CLASS_A_FREQ, CLASS_B_FREQ, CLASS_C_FREQ, CLASS_D_FREQ][class_type]
        expected_spikes = int(frequency * duration_us / 1000000) + 5  # 添加5个额外脉冲
        target_spikes = max(min_spikes_needed, expected_spikes)
        
        # 生成额外的均匀分布脉冲
        import random
        random.seed(42 + pe_id + class_type * 100)
        for i in range(target_spikes - len(spikes)):
            spike_time = start_time_us + random.uniform(0, duration_us * 0.9)
            spikes.append(spike_time)
    
    # 排序脉冲时间
    spikes.sort()
    
    # 写入文件
    with open(filename, 'w') as f:
        f.write("# 神经元ID 时间戳(us) - 4类复杂分类任务\n")
        class_names = ['A', 'B', 'C', 'D']
        f.write(f"# PE{pe_id} - 类别{class_names[class_type]}\n")
        
        # 为该PE的所有神经元生成脉冲 - 改进分配策略
        start_neuron = pe_id * NEURONS_PER_PE
        
        # 确保每个神经元至少有2个脉冲
        spikes_per_neuron = max(2, len(spikes) // NEURONS_PER_PE)
        
        for neuron_offset in range(NEURONS_PER_PE):
            neuron_id = start_neuron + neuron_offset
            
            # 为每个神经元分配连续的脉冲块，而不是交错分配
            start_idx = neuron_offset * spikes_per_neuron
            end_idx = min(start_idx + spikes_per_neuron, len(spikes))
            neuron_spikes = spikes[start_idx:end_idx]
            
            # 如果剩余脉冲不够，给前几个神经元额外分配
            if neuron_offset < len(spikes) % NEURONS_PER_PE:
                extra_spike_idx = NEURONS_PER_PE * spikes_per_neuron + neuron_offset
                if extra_spike_idx < len(spikes):
                    neuron_spikes.append(spikes[extra_spike_idx])
            
            for spike_time in neuron_spikes:
                timestamp_us = int(spike_time)
                f.write(f"{neuron_id} {timestamp_us}\n")
    
    total_spikes = len(spikes)
    return total_spikes

# === 重新生成脉冲数据 ===
print("🔄 重新生成优化的脉冲数据文件...")

spike_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spike_data")
os.makedirs(spike_dir, exist_ok=True)

spike_data_files = []
total_events = 0

# 为输入层(PE 0-3)创建4类复杂数据
for pe_id in INPUT_LAYER:
    # 4类分类：每个PE对应一个类别
    class_type = pe_id  # PE0->A, PE1->B, PE2->C, PE3->D
    class_names = ['A', 'B', 'C', 'D']
    class_name = class_names[class_type]
    
    spike_file = os.path.join(spike_dir, f"complex_input_pe_{pe_id}_class_{class_name}.txt")
    spike_count = create_complex_spike_data(spike_file, pe_id, class_type)
    spike_data_files.append(spike_file)
    total_events += spike_count
    
    freqs = [CLASS_A_FREQ, CLASS_B_FREQ, CLASS_C_FREQ, CLASS_D_FREQ]
    freq = freqs[class_type]
    print(f"  ✅ PE{pe_id}: 类别{class_name} ({freq}Hz), {spike_count}个脉冲")

print(f"\n📊 生成统计:")
print(f"  总文件数: {len(spike_data_files)}")
print(f"  总脉冲数: {total_events}")
print(f"  平均每PE: {total_events // len(INPUT_LAYER)}个脉冲")
print(f"  文件位置: {spike_dir}")

# 验证文件内容
print(f"\n🔍 文件内容验证:")
for i, spike_file in enumerate(spike_data_files):
    with open(spike_file, 'r') as f:
        lines = f.readlines()
    data_lines = [line for line in lines if not line.startswith('#')]
    print(f"  PE{i}: {len(data_lines)}行脉冲事件")

print(f"\n✅ 脉冲数据重新生成完成！")