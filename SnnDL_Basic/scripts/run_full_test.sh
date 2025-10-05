#!/bin/bash

# 优化神经网络分类系统 - 一键运行脚本
# ===========================================

echo "🚀 开始运行优化神经网络分类系统测试"
echo "======================================"

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

# 检查权重文件
WEIGHT_COUNT=$(ls weights/classification_weights_pe_*.bin 2>/dev/null | wc -l)
if [ "$WEIGHT_COUNT" -ne 16 ]; then
    echo "⚠️  权重文件不完整 (找到 $WEIGHT_COUNT 个，期望 16 个)"
    echo "🔧 重新生成权重文件..."
    
    if ! python3 scripts/generate_optimized_weights.py; then
        echo "❌ 权重文件生成失败"
        exit 1
    fi
    
    # 复制新生成的权重文件
    cp /home/anarchy/SST/snnDL_core_system_v2/datasets_classification/classification_weights_pe_*.bin weights/
    echo "✅ 权重文件重新生成完成"
else
    echo "✅ 权重文件检查通过 ($WEIGHT_COUNT 个文件)"
fi

# 检查脉冲数据文件
SPIKE_COUNT=$(ls spike_data/complex_input_pe_*.txt 2>/dev/null | wc -l)
if [ "$SPIKE_COUNT" -ne 4 ]; then
    echo "❌ 脉冲数据文件不完整 (找到 $SPIKE_COUNT 个，期望 4 个)"
    exit 1
fi

echo "✅ 脉冲数据文件检查通过 ($SPIKE_COUNT 个文件)"

# 修改测试脚本中的路径
echo "🔧 配置文件路径..."

# 创建临时测试脚本，修改路径指向当前包目录
sed "s|/home/anarchy/SST/snnDL_core_system_v2/datasets_classification|$PACKAGE_DIR/weights|g" scripts/test_classification_4x4.py > temp_test.py

# 同时需要修改脉冲数据路径
sed -i "s|spike_data_files.append(spike_file)|spike_data_files.append(\"$PACKAGE_DIR/spike_data/complex_input_pe_{}_class_{}.txt\".format(pe_id, class_name))|g" temp_test.py

echo "🧪 开始运行分类测试..."
echo "预计运行时间: ~60秒"
echo ""

# 运行测试，添加超时保护
if timeout 120s sst temp_test.py; then
    echo ""
    echo "✅ 分类测试完成"
else
    echo ""
    echo "⚠️  测试超时或异常结束"
fi

# 清理临时文件
rm -f temp_test.py

# 检查是否有统计文件生成
if [ -f "complex_classification_stats.csv" ]; then
    echo "📊 统计文件已生成: complex_classification_stats.csv"
    mv complex_classification_stats.csv analysis/
fi

# 运行结果分析
echo ""
echo "📈 分析测试结果..."

if python3 scripts/analyze_classification_results.py; then
    echo "✅ 结果分析完成"
    
    # 移动分析结果到analysis目录
    if [ -f "classification_analysis_results.json" ]; then
        mv classification_analysis_results.json analysis/
        echo "📁 分析结果已保存到: analysis/classification_analysis_results.json"
    fi
else
    echo "⚠️  结果分析异常"
fi

# 显示快速统计
echo ""
echo "📋 快速统计:"
echo "============"

if [ -f "analysis/classification_analysis_results.json" ]; then
    echo "📊 详细分析结果请查看: analysis/classification_analysis_results.json"
fi

echo ""
echo "🎉 测试流程完成！"
echo ""
echo "💡 下一步操作:"
echo "   - 查看详细结果: cat analysis/classification_analysis_results.json"
echo "   - 检查权重配置: python3 scripts/inspect_weights.py"
echo "   - 重新运行测试: sst scripts/test_classification_4x4.py"
echo ""