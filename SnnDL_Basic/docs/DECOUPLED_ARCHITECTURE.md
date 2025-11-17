# 解耦架构设计文档

## 🏗️ 三组件解耦设计

本系统采用完全解耦的三组件架构，实现模块化和可维护性：

### 📦 组件架构

```
┌─────────────────────────────────────────────────────┐
│                核心SST仿真配置                        │
│            test_classification_4x4.py              │
│  ┌─────────────────────────────────────────────────┐ │
│  │ • 网络拓扑配置 (4x4 mesh)                        │ │
│  │ • 神经元参数配置 (LIF模型)                       │ │
│  │ • 内存系统配置 (L1缓存+内存控制器)               │ │
│  │ • 网络通信配置 (Merlin网络)                      │ │
│  │ • 仿真参数配置 (时间、统计)                      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   脉冲源生成     │ │   权重文件生成   │ │   结果分析      │
│                │ │                │ │                │
│ generate_spike_ │ │ generate_optim- │ │ analyze_classi- │
│ data.py        │ │ ized_weights.py │ │ fication_results│
│                │ │                │ │ .py            │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │• 脉冲模式   │ │ │ │• 权重矩阵   │ │ │ │• 活动统计   │ │
│ │• 频率编码   │ │ │ │• 连接策略   │ │ │ │• 性能分析   │ │
│ │• 时间分布   │ │ │ │• 分层设计   │ │ │ │• 分类评估   │ │
│ │• 神经元分配 │ │ │ │• 竞争机制   │ │ │ │• 结果可视化 │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
            │               │               │
            ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  spike_data/    │ │    weights/     │ │   analysis/     │
│                │ │                │ │                │
│ • PE0_class_A   │ │ • pe_0.bin     │ │ • results.json  │
│ • PE1_class_B   │ │ • pe_1.bin     │ │ • stats.csv     │
│ • PE2_class_C   │ │ • ...          │ │ • reports/      │
│ • PE3_class_D   │ │ • pe_15.bin    │ │                │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 🔄 工作流程

### 1. 独立组件生成
```bash
# 第一步：生成脉冲源数据
python3 scripts/generate_spike_data.py

# 第二步：生成权重文件  
python3 scripts/generate_optimized_weights.py

# 第三步：运行核心仿真
sst scripts/test_classification_4x4.py

# 第四步：分析结果
python3 scripts/analyze_classification_results.py
```

### 2. 一键集成运行
```bash
# 使用集成脚本
./scripts/run_full_test.sh
```

## 📂 文件依赖关系

### 核心配置脚本 (test_classification_4x4.py)
```python
# 输入依赖
依赖 → spike_data/complex_input_pe_{0-3}_class_{A-D}.txt
依赖 → weights/classification_weights_pe_{0-15}.bin

# 配置功能
✓ 网络拓扑 (4x4 mesh, 16个PE)
✓ 神经元参数 (LIF模型, 阈值配置)
✓ 内存架构 (L1缓存, 内存控制器, WeightLoader)
✓ 网络通信 (Merlin路由器, NIC接口)
✓ 统计收集 (CSV输出, 性能监控)

# 输出产生  
产生 → complex_classification_stats.csv
产生 → 仿真日志和调试信息
```

### 脉冲源生成器 (generate_spike_data.py)
```python
# 生成策略
✓ 4类频率模式 (40Hz, 80Hz, 120Hz, 200Hz)
✓ 时间编码模式 (规律、突发、混合、稀疏)
✓ 神经元分配策略 (每神经元≥2脉冲)
✓ 时间分布优化 (180μs持续时间)

# 输出文件
产生 → spike_data/complex_input_pe_0_class_A.txt (20个脉冲)
产生 → spike_data/complex_input_pe_1_class_B.txt (20个脉冲)
产生 → spike_data/complex_input_pe_2_class_C.txt (20个脉冲)  
产生 → spike_data/complex_input_pe_3_class_D.txt (20个脉冲)
```

### 权重文件生成器 (generate_optimized_weights.py)
```python
# 连接策略
✓ 选择性连接 (输入层→隐藏层1, 隐藏层1→隐藏层2)
✓ 竞争性连接 (隐藏层2→输出层)
✓ 权重强度递增 (10.0 → 12.0 → 15.0)
✓ 稀疏性控制 (75%零权重)

# 输出文件
产生 → weights/classification_weights_pe_{0-15}.bin (16个文件)
每文件 → 4096个float32权重 (16×256矩阵)
```

### 结果分析器 (analyze_classification_results.py)
```python
# 输入依赖
依赖 → complex_classification_stats.csv

# 分析功能
✓ 层级活动统计 (输入/隐藏/输出层)
✓ 网络通信分析 (包计数, 效率计算)
✓ 分类性能评估 (激活模式, 类别区分)
✓ SpikeSource统计 (事件加载/发送率)

# 输出产生
产生 → classification_analysis_results.json
```

## 🎯 解耦优势

### 1. 模块化开发
- **独立测试**: 每个组件可单独测试和验证
- **并行开发**: 不同组件可由不同开发者并行开发
- **版本控制**: 组件版本可独立管理

### 2. 灵活配置
- **脉冲模式**: 可生成不同频率和模式的脉冲数据
- **权重策略**: 可实验不同的连接和权重分布
- **网络架构**: 核心配置可调整网络规模和参数

### 3. 可扩展性
- **新脉冲模式**: 添加新的编码策略无需修改核心配置
- **新权重算法**: 实验新的权重生成算法
- **新分析方法**: 添加新的性能分析指标

### 4. 可重用性
- **脉冲数据**: 同一套脉冲数据可用于不同网络配置
- **权重文件**: 可在不同仿真参数下重用权重
- **配置模板**: 核心配置可作为其他项目模板

## 🔧 自定义扩展

### 添加新脉冲模式
```python
# 在 generate_spike_data.py 中添加
elif class_type == 4:  # 新类别E
    # 自定义脉冲生成逻辑
    pass
```

### 添加新权重策略
```python
# 在 generate_optimized_weights.py 中添加
elif connection_type == "custom":
    # 自定义权重生成逻辑
    pass
```

### 修改网络架构
```python
# 在 test_classification_4x4.py 中修改
MESH_SIZE = 6  # 扩展到6x6网络
# 相应调整其他参数
```

## 📊 性能基准

### 组件执行时间
- **脉冲数据生成**: ~1秒
- **权重文件生成**: ~2秒  
- **核心仿真**: ~60秒
- **结果分析**: ~1秒

### 资源消耗
- **脉冲数据**: 4个文件, ~4KB
- **权重文件**: 16个文件, ~256KB
- **仿真内存**: ~1GB
- **结果文件**: ~100KB

这种解耦设计确保了系统的模块化、可维护性和可扩展性，同时保持了高性能。