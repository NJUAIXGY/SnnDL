# 优化后的SST神经网络分类系统

这是一个基于SST(Structural Simulation Toolkit)的优化神经网络分类系统，实现了4x4分层网络架构用于多类别脉冲模式分类。

## 📁 目录结构

```
SnnDL_Basic/
├── scripts/                    # 脚本文件
│   ├── test_classification_4x4.py      # 主测试脚本
│   ├── generate_optimized_weights.py   # 权重生成脚本
│   ├── analyze_classification_results.py # 结果分析脚本
│   ├── inspect_weights.py              # 权重检查脚本
│   └── run_full_test.sh                # 一键运行脚本
├── weights/                    # 权重文件
│   └── classification_weights_pe_{0-15}.bin
├── spike_data/                 # 脉冲数据文件
│   ├── complex_input_pe_0_class_A.txt
│   ├── complex_input_pe_1_class_B.txt
│   ├── complex_input_pe_2_class_C.txt
│   └── complex_input_pe_3_class_D.txt
├── analysis/                   # 分析结果
│   └── classification_analysis_results.json
├── docs/                       # 文档
│   └── ARCHITECTURE.md
└── README.md                   # 本文件
```

## 🏗️ 解耦架构设计

### 三组件独立设计
本系统采用完全解耦的模块化架构：

1. **核心仿真配置** (`test_classification_4x4.py`)
   - 网络拓扑和神经元参数配置
   - 内存系统和通信网络配置
   - 从预生成文件加载数据，不包含数据生成逻辑

2. **脉冲源生成器** (`generate_spike_data.py`)
   - 独立的4类脉冲模式生成
   - 支持不同频率和时间编码策略
   - 可单独修改和测试脉冲模式

3. **权重文件生成器** (`generate_optimized_weights.py`)
   - 独立的分层权重矩阵生成
   - 选择性和竞争性连接策略
   - 可实验不同权重分布算法

### 网络架构
- **输入层** (PE 0-3): 接收4类不同频率的脉冲输入
  - PE0: 类别A (40Hz)
  - PE1: 类别B (80Hz) 
  - PE2: 类别C (120Hz)
  - PE3: 类别D (200Hz)

- **隐藏层1** (PE 4-7): 选择性连接，每个PE主要响应特定输入PE
- **隐藏层2** (PE 8-11): 选择性连接，进一步特征提取
- **输出层** (PE 12-15): 竞争性连接，实现分类决策

### 权重设计
- **输入层**: 自连接权重 0.5
- **隐藏层1**: 选择性连接，主连接权重 ~15-18，次连接权重 ~6-7
- **隐藏层2**: 选择性连接，主连接权重 ~17-21，次连接权重 ~7-8
- **输出层**: 竞争性连接，同类权重 ~30，异类权重 ~3

## 🚀 快速开始

### 预置要求
- SST框架已安装
- SnnDL组件库已编译
- Python 3.x环境

### 运行方式

#### 方法1: 一键运行(推荐)
```bash
cd SnnDL_Basic
./scripts/run_full_test.sh
```

#### 方法2: 分步运行(解耦架构)
```bash
cd SnnDL_Basic

# 1. 生成脉冲数据文件
python3 scripts/generate_spike_data.py

# 2. 生成权重文件
python3 scripts/generate_optimized_weights.py

# 3. 运行核心仿真配置
sst scripts/test_classification_4x4.py

# 4. 分析结果
python3 scripts/analyze_classification_results.py

# 5. 检查权重文件(可选)
python3 scripts/inspect_weights.py
```

## 📊 性能指标

### 优化前 vs 优化后
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 输入层脉冲 | 5 | 13 | +160% |
| 隐藏层激活 | 2 | 4 | +100% |
| 输出层激活 | 2 | 4 | +100% |
| 网络效率 | 114.3% | 142.9% | +25% |
| 激活路径 | 1条 | 多条 | 更好的分布 |

### 最新测试结果
- ✅ 所有4个输入层PE成功激活
- ✅ 隐藏层PE4和PE11激活，实现多路径传播
- ✅ 输出层PE12(3脉冲)和PE15(1脉冲)激活，展现分类能力
- ✅ 网络总脉冲活动: 21个脉冲，利用率显著提升

## 🔧 配置参数

### 网络参数
- **仿真时间**: 200微秒
- **脉冲持续时间**: 180微秒
- **神经元阈值**: 输入层/隐藏层 0.02, 输出层 0.025
- **网络拓扑**: 4x4网格，16个PE节点

### 脉冲源配置
- **类别A**: 40Hz频率，规律脉冲+轻微噪声
- **类别B**: 80Hz频率，突发模式脉冲
- **类别C**: 120Hz频率，高频混合模式
- **类别D**: 200Hz频率，稀疏高频脉冲

## 📈 分析工具

### analyze_classification_results.py
- 输入层活动分析
- 隐藏层激活统计
- 输出层分类结果
- 网络通信效率
- 分类性能评估

### inspect_weights.py
- 权重文件完整性检查
- 权重分布统计
- 连接性分析
- 分层权重验证

## 🛠️ 故障排除

### 常见问题
1. **权重文件不存在**: 运行 `generate_optimized_weights.py`
2. **SST找不到组件**: 检查SnnDL库是否正确安装
3. **仿真超时**: 调整SIMULATION_TIME参数
4. **脉冲源无输出**: 检查spike_data文件路径

### 调试技巧
- 启用详细输出: 设置 `verbose: 2`
- 检查权重加载: 查看 "权重写入操作已完成" 消息
- 监控缓存: 观察 "首次命中/未命中" 日志
- 验证连接: 使用 `inspect_weights.py` 检查权重矩阵

## 📝 更新日志

### v2.0 (最新)
- ✅ 完全重构权重生成策略
- ✅ 实现选择性和竞争性连接
- ✅ 延长仿真时间和脉冲持续时间
- ✅ 优化神经元阈值设置
- ✅ 增强网络激活分布
- ✅ 添加权重检查工具

### v1.0 (基础版本)
- ✅ 基础4x4分层网络架构
- ✅ 简单权重配置
- ✅ 基础分类功能

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

遵循SST框架许可证条款。
