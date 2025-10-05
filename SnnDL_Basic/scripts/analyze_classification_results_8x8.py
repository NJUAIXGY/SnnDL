#!/usr/bin/env python3

import pandas as pd
import json

def analyze_classification_results_8x8():
    """分析8x8分层网络分类任务结果"""
    
    # 8x8网络分层定义
    INPUT_LAYER = list(range(0, 8))      # PE 0-7: 输入层 (8个PE)
    HIDDEN_LAYER_1 = list(range(8, 24))   # PE 8-23: 隐藏层1 (16个PE)
    HIDDEN_LAYER_2 = list(range(24, 40))  # PE 24-39: 隐藏层2 (16个PE)
    HIDDEN_LAYER_3 = list(range(40, 56))  # PE 40-55: 隐藏层3 (16个PE)
    OUTPUT_LAYER = list(range(56, 64))   # PE 56-63: 输出层 (8个PE)
    
    # 8类别定义
    CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    FREQUENCIES = [40, 60, 80, 100, 120, 150, 180, 200]
    
    print("🧠 8x8分层神经网络分类任务结果分析")
    print("=" * 60)
    
    try:
        # 读取统计数据
        df = pd.read_csv("complex_classification_8x8_stats.csv")
        
        # 分析输入层活动
        print("📊 输入层活动分析 (8个PE，8个类别):")
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
            
            class_name = CLASS_NAMES[pe_id]
            frequency = FREQUENCIES[pe_id]
            input_stats[pe_id] = {
                'class': class_name,
                'frequency': frequency,
                'processed': processed,
                'received': received,
                'sent': sent
            }
            
            print(f"  PE{pe_id} (类别{class_name}, {frequency}Hz): 接收{received}, 处理{processed}, 发送{sent}脉冲")
        
        # 分析隐藏层活动
        print("\n🔍 隐藏层活动分析:")
        
        def analyze_hidden_layer(layer_name, layer_pes):
            layer_stats = {}
            total_processed = 0
            total_sent = 0
            active_pes = 0
            
            for pe_id in layer_pes:
                pe_name = f"multicore_pe_{pe_id}"
                spikes_processed = df[(df['ComponentName'] == pe_name) & 
                                    (df['StatisticName'] == 'total_spikes_processed')]['Sum.u64'].values
                spikes_sent = df[(df['ComponentName'] == pe_name) & 
                               (df['StatisticName'] == 'external_spikes_sent')]['Sum.u64'].values
                
                processed = spikes_processed[0] if len(spikes_processed) > 0 else 0
                sent = spikes_sent[0] if len(spikes_sent) > 0 else 0
                
                layer_stats[pe_id] = {'processed': processed, 'sent': sent}
                total_processed += processed
                total_sent += sent
                if processed > 0:
                    active_pes += 1
            
            activation_rate = (active_pes / len(layer_pes)) * 100
            print(f"  {layer_name}: 总处理{total_processed}, 总发送{total_sent}脉冲")
            print(f"    激活PE: {active_pes}/{len(layer_pes)} ({activation_rate:.1f}%)")
            
            # 显示最活跃的前5个PE
            sorted_pes = sorted(layer_stats.items(), key=lambda x: x[1]['processed'], reverse=True)[:5]
            if sorted_pes[0][1]['processed'] > 0:
                print(f"    最活跃PE: ", end="")
                for pe_id, stats in sorted_pes:
                    if stats['processed'] > 0:
                        print(f"PE{pe_id}({stats['processed']})", end=" ")
                print()
            
            return layer_stats, total_processed, total_sent
        
        hidden1_stats, hidden1_activity, hidden1_sent = analyze_hidden_layer("隐藏层1 (PE 8-23)", HIDDEN_LAYER_1)
        hidden2_stats, hidden2_activity, hidden2_sent = analyze_hidden_layer("隐藏层2 (PE 24-39)", HIDDEN_LAYER_2)
        hidden3_stats, hidden3_activity, hidden3_sent = analyze_hidden_layer("隐藏层3 (PE 40-55)", HIDDEN_LAYER_3)
        
        # 分析输出层活动（关键分类结果）
        print("\n🎯 输出层活动分析（分类结果）:")
        output_stats = {}
        classification_results = {}
        
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
            
            expected_class = CLASS_NAMES[pe_id - min(OUTPUT_LAYER)]  # PE56->A, PE57->B, ...
            
            output_stats[pe_id] = {
                'expected_class': expected_class,
                'processed': processed,
                'received': received,
                'utilization': utilization
            }
            
            classification_results[expected_class] = processed
            
            print(f"  输出PE{pe_id} (期望类别{expected_class}): 接收{received}, 处理{processed}脉冲, 利用率{utilization:.4f}")
        
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
            
            class_name = CLASS_NAMES[pe_id]
            frequency = FREQUENCIES[pe_id]
            spike_source_stats[pe_id] = {
                'class': class_name,
                'frequency': frequency,
                'loaded': loaded,
                'sent': sent
            }
            
            print(f"  SpikeSource{pe_id} (类别{class_name}, {frequency}Hz): 加载{loaded}, 发送{sent}事件")
        
        # 分析网络通信
        print("\n🌐 网络通信分析:")
        total_packets_sent = 0
        total_packets_received = 0
        layer_communications = {'input': 0, 'hidden1': 0, 'hidden2': 0, 'hidden3': 0, 'output': 0}
        
        for pe_id in range(64):
            pe_name = f"multicore_pe_{pe_id}"
            sent = df[(df['ComponentName'] == pe_name) & 
                     (df['StatisticName'] == 'external_spikes_sent')]['Sum.u64'].values
            received = df[(df['ComponentName'] == pe_name) & 
                         (df['StatisticName'] == 'external_spikes_received')]['Sum.u64'].values
            
            sent_count = sent[0] if len(sent) > 0 else 0
            received_count = received[0] if len(received) > 0 else 0
            
            total_packets_sent += sent_count
            total_packets_received += received_count
            
            # 分层统计通信量
            if pe_id in INPUT_LAYER:
                layer_communications['input'] += sent_count
            elif pe_id in HIDDEN_LAYER_1:
                layer_communications['hidden1'] += sent_count
            elif pe_id in HIDDEN_LAYER_2:
                layer_communications['hidden2'] += sent_count
            elif pe_id in HIDDEN_LAYER_3:
                layer_communications['hidden3'] += sent_count
            elif pe_id in OUTPUT_LAYER:
                layer_communications['output'] += sent_count
        
        print(f"  总发送包数: {total_packets_sent}")
        print(f"  总接收包数: {total_packets_received}")
        print(f"  网络效率: {(total_packets_received/total_packets_sent*100):.1f}%" if total_packets_sent > 0 else "  网络效率: N/A")
        
        print(f"\n  分层通信统计:")
        print(f"    输入层发送: {layer_communications['input']}包")
        print(f"    隐藏层1发送: {layer_communications['hidden1']}包")
        print(f"    隐藏层2发送: {layer_communications['hidden2']}包")
        print(f"    隐藏层3发送: {layer_communications['hidden3']}包")
        print(f"    输出层发送: {layer_communications['output']}包")
        
        # 分类性能评估
        print("\n🏆 8类分类性能评估:")
        
        # 计算各层的总活动
        input_activity = sum([stats['processed'] for stats in input_stats.values()])
        output_activity = sum([stats['processed'] for stats in output_stats.values()])
        
        print(f"  输入层总活动: {input_activity}脉冲")
        print(f"  隐藏层1总活动: {hidden1_activity}脉冲")
        print(f"  隐藏层2总活动: {hidden2_activity}脉冲")
        print(f"  隐藏层3总活动: {hidden3_activity}脉冲")
        print(f"  输出层总活动: {output_activity}脉冲")
        
        # 信息传播效率
        if input_activity > 0:
            propagation_efficiency = (output_activity / input_activity) * 100
            print(f"  信息传播效率: {propagation_efficiency:.1f}% (输出/输入)")
        
        # 分析输出层激活模式
        print(f"\n🎯 输出层分类激活模式:")
        sorted_outputs = sorted(classification_results.items(), key=lambda x: x[1], reverse=True)
        
        for i, (class_name, activation) in enumerate(sorted_outputs):
            if activation > 0:
                rank = i + 1
                print(f"  第{rank}名: 类别{class_name} ({activation}次激活)")
        
        # 判断分类成功情况
        max_class = max(classification_results, key=classification_results.get)
        max_activation = classification_results[max_class]
        
        if max_activation > 0:
            print(f"\n✅ 网络成功产生分类输出")
            print(f"  主导分类: 类别{max_class} ({max_activation}次激活)")
            
            # 计算分类置信度
            total_output = sum(classification_results.values())
            if total_output > 0:
                confidence = (max_activation / total_output) * 100
                print(f"  分类置信度: {confidence:.1f}%")
        else:
            print(f"\n⚠️ 输出层未产生分类激活")
        
        # 多样性分析
        active_output_classes = sum(1 for activation in classification_results.values() if activation > 0)
        print(f"  激活类别数: {active_output_classes}/8")
        
        # 保存分析结果
        analysis_results = {
            'network_architecture': {
                'input_layer': len(INPUT_LAYER),
                'hidden_layer_1': len(HIDDEN_LAYER_1),
                'hidden_layer_2': len(HIDDEN_LAYER_2),
                'hidden_layer_3': len(HIDDEN_LAYER_3),
                'output_layer': len(OUTPUT_LAYER),
                'total_pes': 64
            },
            'input_layer': input_stats,
            'hidden_layer_1': hidden1_stats,
            'hidden_layer_2': hidden2_stats,
            'hidden_layer_3': hidden3_stats,
            'output_layer': output_stats,
            'spike_sources': spike_source_stats,
            'network_stats': {
                'total_sent': total_packets_sent,
                'total_received': total_packets_received,
                'efficiency': total_packets_received/total_packets_sent if total_packets_sent > 0 else 0,
                'layer_communications': layer_communications
            },
            'layer_activities': {
                'input': input_activity,
                'hidden1': hidden1_activity,
                'hidden2': hidden2_activity,
                'hidden3': hidden3_activity,
                'output': output_activity
            },
            'classification_results': classification_results,
            'performance_metrics': {
                'propagation_efficiency': propagation_efficiency if input_activity > 0 else 0,
                'active_output_classes': active_output_classes,
                'dominant_class': max_class if max_activation > 0 else None,
                'classification_confidence': (max_activation / total_output * 100) if total_output > 0 else 0
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
        
        with open('classification_analysis_8x8_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 8x8网络分析结果已保存到 classification_analysis_8x8_results.json")
        
        # 总结报告
        print(f"\n📋 8x8网络性能总结:")
        print(f"  🔥 输入激活: {input_activity}次 (8个PE)")
        print(f"  🧠 隐藏层处理: {hidden1_activity + hidden2_activity + hidden3_activity}次 (48个PE)")
        print(f"  🎯 输出激活: {output_activity}次 (8个PE)")
        print(f"  📡 网络效率: {(total_packets_received/total_packets_sent*100):.1f}%")
        print(f"  🏆 分类成功: {'是' if max_activation > 0 else '否'}")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_classification_results_8x8()