#!/bin/bash

# 8x8优化神经网络分类系统 - 一键运行脚本
# ===============================================

echo "🚀 开始运行8x8优化神经网络分类系统测试"
echo "========================================"

# 检查工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

echo "📁 工作目录: $PACKAGE_DIR"
cd "$PACKAGE_DIR"

# 检查SST是否可用
if ! command -v sst &> /dev/null; then
    echo "❌ 错误: SST命令不可用，请检查SST安装"
    exit 1
fi

echo "✅ SST环境检查通过"

# 创建8x8数据目录
mkdir -p weights_8x8 spike_data_8x8 analysis_8x8

# 检查或生成脉冲数据文件
SPIKE_COUNT=$(ls spike_data_8x8/complex_input_pe_*.txt 2>/dev/null | wc -l)
if [ "$SPIKE_COUNT" -ne 8 ]; then
    echo "⚠️  脉冲数据文件不完整 (找到 $SPIKE_COUNT 个，期望 8 个)"
    echo "🔧 生成脉冲数据文件..."
    
    if ! python3 scripts/generate_spike_data_8x8.py; then
        echo "❌ 脉冲数据生成失败"
        exit 1
    fi
    
    echo "✅ 脉冲数据文件生成完成"
else
    echo "✅ 脉冲数据文件检查通过 ($SPIKE_COUNT 个文件)"
fi

# 检查或生成权重文件
WEIGHT_COUNT=$(ls weights_8x8/classification_weights_pe_*.bin 2>/dev/null | wc -l)
if [ "$WEIGHT_COUNT" -ne 64 ]; then
    echo "⚠️  权重文件不完整 (找到 $WEIGHT_COUNT 个，期望 64 个)"
    echo "🔧 生成权重文件..."
    
    if ! python3 scripts/generate_optimized_weights_8x8.py; then
        echo "❌ 权重文件生成失败"
        exit 1
    fi
    
    echo "✅ 权重文件生成完成"
else
    echo "✅ 权重文件检查通过 ($WEIGHT_COUNT 个文件)"
fi

echo "🧪 开始运行8x8分类测试..."
echo "预计运行时间: ~120秒 (8x8网络需要更长时间)"
echo "网络规模: 64个PE, 1024个神经元, 5层架构"
echo ""

# 运行测试，增加超时时间以适应更大网络
if timeout 300s sst scripts/test_classification_8x8.py; then
    echo ""
    echo "✅ 8x8分类测试完成"
else
    echo ""
    echo "⚠️  测试超时或异常结束"
fi

# 检查是否有统计文件生成
if [ -f "complex_classification_8x8_stats.csv" ]; then
    echo "📊 统计文件已生成: complex_classification_8x8_stats.csv"
    mv complex_classification_8x8_stats.csv analysis_8x8/
fi

# 运行结果分析
echo ""
echo "📈 分析8x8测试结果..."

if python3 scripts/analyze_classification_results_8x8.py; then
    echo "✅ 结果分析完成"
    
    # 移动分析结果到analysis目录
    if [ -f "classification_analysis_8x8_results.json" ]; then
        mv classification_analysis_8x8_results.json analysis_8x8/
        echo "📁 分析结果已保存到: analysis_8x8/classification_analysis_8x8_results.json"
    fi
else
    echo "⚠️  结果分析异常"
fi

# 显示8x8网络统计
echo ""
echo "📋 8x8网络快速统计:"
echo "===================="

if [ -f "analysis_8x8/classification_analysis_8x8_results.json" ]; then
    echo "📊 详细分析结果请查看: analysis_8x8/classification_analysis_8x8_results.json"
    
    # 尝试提取关键统计信息
    if command -v jq &> /dev/null; then
        echo ""
        echo "🔍 关键性能指标:"
        
        # 提取输入层活动
        input_activity=$(jq -r '.layer_activities.input // 0' analysis_8x8/classification_analysis_8x8_results.json)
        echo "  输入层活动: $input_activity 次"
        
        # 提取网络效率
        network_efficiency=$(jq -r '.network_stats.efficiency // 0' analysis_8x8/classification_analysis_8x8_results.json)
        echo "  网络效率: $(echo "scale=1; $network_efficiency * 100" | bc -l)%"
        
        # 提取主导分类
        dominant_class=$(jq -r '.performance_metrics.dominant_class // "无"' analysis_8x8/classification_analysis_8x8_results.json)
        echo "  主导分类: $dominant_class"
        
        # 提取激活类别数
        active_classes=$(jq -r '.performance_metrics.active_output_classes // 0' analysis_8x8/classification_analysis_8x8_results.json)
        echo "  激活类别: $active_classes/8"
        
    fi
fi

echo ""
echo "🎉 8x8网络测试流程完成！"
echo ""
echo "📊 8x8网络特点:"
echo "   - 网络规模: 64个PE (4x4网络的4倍)"
echo "   - 神经元数: 1024个 (4x4网络的4倍)"
echo "   - 分类能力: 8类 (4x4网络的2倍)"
echo "   - 网络深度: 5层 (输入+3隐藏+输出)"
echo ""
echo "💡 下一步操作:"
echo "   - 查看详细结果: cat analysis_8x8/classification_analysis_8x8_results.json"
echo "   - 对比4x4结果: diff analysis/classification_analysis_results.json analysis_8x8/classification_analysis_8x8_results.json"
echo "   - 重新运行测试: sst scripts/test_classification_8x8.py"
echo "   - 检查权重配置: python3 scripts/inspect_weights.py"
echo ""