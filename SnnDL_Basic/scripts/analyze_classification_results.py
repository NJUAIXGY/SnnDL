#!/usr/bin/env python3

import pandas as pd
import json

def analyze_classification_results():
    """分析4x4分层网络分类任务结果"""
    
    # 网络分层定义
    INPUT_LAYER = list(range(0, 4))      # PE 0-3: 输入层
    HIDDEN_LAYER_1 = list(range(4, 8))   # PE 4-7: 隐藏层1
    HIDDEN_LAYER_2 = list(range(8, 12))  # PE 8-11: 隐藏层2
    OUTPUT_LAYER = list(range(12, 16))   # PE 12-15: 输出层
    
    # 类别定义
    CLASS_A_INPUT_PES = [0, 1]  # PE 0-1: 类别A输入
    CLASS_B_INPUT_PES = [2, 3]  # PE 2-3: 类别B输入
    
    print("🧠 4x4分层神经网络分类任务结果分析")
    print("=" * 50)
    
    try:
        # 读取统计数据
        df = pd.read_csv("complex_classification_stats.csv")
        
        # 分析输入层活动
        print("📊 输入层活动分析:")
        input_stats = {}
        
        for pe_id in INPUT_LAYER:
            pe_name = f"multicore_pe_{pe_id}"
            spikes_processed = df[(df['ComponentName'] == pe_name) & 
                                (df['StatisticName'] == 'total_spikes_processed')]['Sum.u64'].values
            spikes_received = df[(df['ComponentName'] == pe_name) & 
                               (df['StatisticName'] == 'external_spikes_received')]['Sum.u64'].values
            spikes_sent = df[(df['ComponentName'] == pe_name) & 
                           (df['StatisticName'] == 'external_spikes_sent')]['Sum.u64'].values
            
            processed = spikes_processed[0] if len(spikes_processed) > 0 else 0
            received = spikes_received[0] if len(spikes_received) > 0 else 0
            sent = spikes_sent[0] if len(spikes_sent) > 0 else 0
            
            class_type = "类别A" if pe_id in CLASS_A_INPUT_PES else "类别B"
            input_stats[pe_id] = {
                'class': class_type,
                'processed': processed,
                'received': received,
                'sent': sent
            }
            
            print(f"  PE{pe_id} ({class_type}): 接收{received}, 处理{processed}, 发送{sent}脉冲")
        
        # 分析隐藏层活动
        print("\n🔍 隐藏层活动分析:")
        hidden1_stats = {}
        hidden2_stats = {}
        
        for pe_id in HIDDEN_LAYER_1:
            pe_name = f"multicore_pe_{pe_id}"
            spikes_processed = df[(df['ComponentName'] == pe_name) & 
                                (df['StatisticName'] == 'total_spikes_processed')]['Sum.u64'].values
            spikes_sent = df[(df['ComponentName'] == pe_name) & 
                           (df['StatisticName'] == 'external_spikes_sent')]['Sum.u64'].values
            
            processed = spikes_processed[0] if len(spikes_processed) > 0 else 0
            sent = spikes_sent[0] if len(spikes_sent) > 0 else 0
            
            hidden1_stats[pe_id] = {'processed': processed, 'sent': sent}
            print(f"  隐藏层1 PE{pe_id}: 处理{processed}, 发送{sent}脉冲")
        
        for pe_id in HIDDEN_LAYER_2:
            pe_name = f"multicore_pe_{pe_id}"
            spikes_processed = df[(df['ComponentName'] == pe_name) & 
                                (df['StatisticName'] == 'total_spikes_processed')]['Sum.u64'].values
            spikes_sent = df[(df['ComponentName'] == pe_name) & 
                           (df['StatisticName'] == 'external_spikes_sent')]['Sum.u64'].values
            
            processed = spikes_processed[0] if len(spikes_processed) > 0 else 0
            sent = spikes_sent[0] if len(spikes_sent) > 0 else 0
            
            hidden2_stats[pe_id] = {'processed': processed, 'sent': sent}
            print(f"  隐藏层2 PE{pe_id}: 处理{processed}, 发送{sent}脉冲")
        
        # 分析输出层活动（关键分类结果）
        print("\n🎯 输出层活动分析（分类结果）:")
        output_stats = {}
        
        for pe_id in OUTPUT_LAYER:
            pe_name = f"multicore_pe_{pe_id}"
            spikes_processed = df[(df['ComponentName'] == pe_name) & 
                                (df['StatisticName'] == 'total_spikes_processed')]['Sum.u64'].values
            spikes_received = df[(df['ComponentName'] == pe_name) & 
                               (df['StatisticName'] == 'external_spikes_received')]['Sum.u64'].values
            core_utilization = df[(df['ComponentName'] == pe_name) & 
                                (df['StatisticName'] == 'avg_core_utilization')]['Sum.f64'].values
            
            processed = spikes_processed[0] if len(spikes_processed) > 0 else 0
            received = spikes_received[0] if len(spikes_received) > 0 else 0
            utilization = core_utilization[0] if len(core_utilization) > 0 else 0.0
            
            output_stats[pe_id] = {
                'processed': processed,
                'received': received,
                'utilization': utilization
            }
            
            print(f"  输出层 PE{pe_id}: 接收{received}, 处理{processed}脉冲, 利用率{utilization:.4f}")
        
        # 分析SpikeSource输入
        print("\n📡 SpikeSource输入分析:")
        spike_source_stats = {}
        
        for pe_id in INPUT_LAYER:
            source_name = f"spike_source_{pe_id}"
            events_loaded = df[(df['ComponentName'] == source_name) & 
                              (df['StatisticName'] == 'events_loaded')]['Sum.u64'].values
            events_sent = df[(df['ComponentName'] == source_name) & 
                            (df['StatisticName'] == 'events_sent')]['Sum.u64'].values
            
            loaded = events_loaded[0] if len(events_loaded) > 0 else 0
            sent = events_sent[0] if len(events_sent) > 0 else 0
            
            class_type = "类别A" if pe_id in CLASS_A_INPUT_PES else "类别B"
            spike_source_stats[pe_id] = {
                'class': class_type,
                'loaded': loaded,
                'sent': sent
            }
            
            print(f"  SpikeSource{pe_id} ({class_type}): 加载{loaded}, 发送{sent}事件")
        
        # 分析网络通信
        print("\n🌐 网络通信分析:")
        total_packets_sent = 0
        total_packets_received = 0
        
        for pe_id in range(16):
            pe_name = f"multicore_pe_{pe_id}"
            sent = df[(df['ComponentName'] == pe_name) & 
                     (df['StatisticName'] == 'external_spikes_sent')]['Sum.u64'].values
            received = df[(df['ComponentName'] == pe_name) & 
                         (df['StatisticName'] == 'external_spikes_received')]['Sum.u64'].values
            
            sent_count = sent[0] if len(sent) > 0 else 0
            received_count = received[0] if len(received) > 0 else 0
            
            total_packets_sent += sent_count
            total_packets_received += received_count
        
        print(f"  总发送包数: {total_packets_sent}")
        print(f"  总接收包数: {total_packets_received}")
        print(f"  网络效率: {(total_packets_received/total_packets_sent*100):.1f}%" if total_packets_sent > 0 else "  网络效率: N/A")
        
        # 分类性能评估
        print("\n🏆 分类性能评估:")
        
        # 计算各层的总活动
        input_activity = sum([stats['processed'] for stats in input_stats.values()])
        hidden1_activity = sum([stats['processed'] for stats in hidden1_stats.values()])
        hidden2_activity = sum([stats['processed'] for stats in hidden2_stats.values()])
        output_activity = sum([stats['processed'] for stats in output_stats.values()])
        
        print(f"  输入层总活动: {input_activity}脉冲")
        print(f"  隐藏层1总活动: {hidden1_activity}脉冲")
        print(f"  隐藏层2总活动: {hidden2_activity}脉冲")
        print(f"  输出层总活动: {output_activity}脉冲")
        
        # 分析是否成功区分类别
        class_a_total_input = sum([input_stats[pe_id]['processed'] for pe_id in CLASS_A_INPUT_PES])
        class_b_total_input = sum([input_stats[pe_id]['processed'] for pe_id in CLASS_B_INPUT_PES])
        
        print(f"\n📈 输入类别分析:")
        print(f"  类别A总输入: {class_a_total_input}脉冲")
        print(f"  类别B总输入: {class_b_total_input}脉冲")
        
        # 输出层激活模式
        print(f"\n🎯 输出层激活模式:")
        max_output_pe = max(output_stats.keys(), key=lambda k: output_stats[k]['processed'])
        max_activity = output_stats[max_output_pe]['processed']
        
        print(f"  最活跃输出PE: PE{max_output_pe} ({max_activity}脉冲)")
        
        if max_activity > 0:
            print("  ✅ 网络成功产生输出激活")
        else:
            print("  ⚠️  输出层未产生激活")
        
        # 保存分析结果
        analysis_results = {
            'input_layer': input_stats,
            'hidden_layer_1': hidden1_stats,
            'hidden_layer_2': hidden2_stats,
            'output_layer': output_stats,
            'spike_sources': spike_source_stats,
            'network_stats': {
                'total_sent': total_packets_sent,
                'total_received': total_packets_received,
                'efficiency': total_packets_received/total_packets_sent if total_packets_sent > 0 else 0
            },
            'layer_activities': {
                'input': input_activity,
                'hidden1': hidden1_activity,
                'hidden2': hidden2_activity,
                'output': output_activity
            }
        }
        
        # 转换numpy int64为普通int
        def convert_int64(obj):
            if hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_int64(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_int64(v) for v in obj]
            return obj
        
        analysis_results = convert_int64(analysis_results)
        
        with open('classification_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 分析结果已保存到 classification_analysis_results.json")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")

if __name__ == "__main__":
    analyze_classification_results()