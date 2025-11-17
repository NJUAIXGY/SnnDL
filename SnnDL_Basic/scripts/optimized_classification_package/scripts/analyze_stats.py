#!/usr/bin/env python3
import csv
import os
import sys

def to_int(row, key):
    try:
        return int(row.get(key, '0'))
    except Exception:
        return 0

def main():
    # 默认统计文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, 'complex_classification_stats_8x8.csv'),      # 新：8x8 扩展默认
        os.path.join(base_dir, 'complex_classification_stats.csv'),           # 常见：在 optimized_classification_package/
        os.path.join(os.path.dirname(base_dir), 'complex_classification_stats_8x8.csv'),
        os.path.join(os.path.dirname(base_dir), 'complex_classification_stats.csv'),  # 备选：在仓库根目录
    ]
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if not os.path.exists(path):
        print("未找到统计文件，请指定路径。例如：")
        print("  python3 scripts/analyze_stats.py /path/to/complex_classification_stats.csv")
        sys.exit(1)

    # 汇总容器
    router_packets = 0
    router_bits = 0
    pe_routes_entries_sum = 0
    fanout_sum = 0
    fanout_count = 0
    fanout_max = 0
    pe_spikes_sent = 0
    pe_spikes_recv = 0

    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('StatisticName', '')
            comp = row.get('ComponentName', '')

            # Merlin router 聚合
            if comp.startswith('router_'):
                if name == 'send_packet_count':
                    router_packets += to_int(row, 'Sum.u64')
                elif name == 'send_bit_count':
                    router_bits += to_int(row, 'Sum.u64')

            # SnnPESubComponent 自定义统计
            if name == 'routes_entries':
                pe_routes_entries_sum += to_int(row, 'Sum.u64')
            elif name == 'fanout_per_spike':
                fanout_sum += to_int(row, 'Sum.u64')
                fanout_count += to_int(row, 'Count.u64')
                fanout_max = max(fanout_max, to_int(row, 'Max.u64'))

            # MultiCorePE 关键指标（若存在）
            if name == 'total_network_spikes_sent':
                pe_spikes_sent += to_int(row, 'Sum.u64')
            elif name == 'total_external_spikes_received':
                pe_spikes_recv += to_int(row, 'Sum.u64')

    print('=== 统计汇总 ===')
    print(f'- Router 发送包总数: {router_packets}')
    print(f'- Router 发送比特总数: {router_bits}')
    print(f'- 路由表条目总和(routes_entries): {pe_routes_entries_sum}')
    if fanout_count > 0:
        avg_fanout = fanout_sum / fanout_count
        print(f'- 每次发放扇出(fanout_per_spike): 平均={avg_fanout:.2f}, 峰值={fanout_max}')
    else:
        print(f'- 每次发放扇出(fanout_per_spike): 暂无记录')
    print(f'- PE 发送网络脉冲总数(total_network_spikes_sent): {pe_spikes_sent}')
    print(f'- PE 接收外部脉冲总数(total_external_spikes_received): {pe_spikes_recv}')

if __name__ == '__main__':
    main()
