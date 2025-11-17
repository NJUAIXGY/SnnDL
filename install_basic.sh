#!/bin/bash
# SnnDL基本库安装脚本

echo "正在安装SnnDL基本库..."

# 确保在正确的目录
cd "$(dirname "$0")"

# 编译基本库
echo "编译基本库..."
g++ -Wall -Wextra -std=c++17 -fPIC -shared \
    -I/home/anarchy/SST/sst_install/include \
    -I/home/anarchy/SST/sst_install/include/sst \
    SnnPE.cc SpikeSource.cc \
    -o libSnnDL_basic.so

if [ $? -eq 0 ]; then
    echo "✅ 编译成功"
else
    echo "❌ 编译失败"
    exit 1
fi

# 复制到SST库目录
echo "安装库文件..."
cp libSnnDL_basic.so /home/anarchy/SST/sst_install/lib/sst-elements-library/
if [ $? -eq 0 ]; then
    echo "✅ 库安装成功"
else
    echo "❌ 库安装失败"
    exit 1
fi

# 复制头文件（可选，用于其他组件引用）
echo "安装头文件..."
mkdir -p /home/anarchy/SST/sst_install/include/sst/elements/SnnDL/
cp *.h /home/anarchy/SST/sst_install/include/sst/elements/SnnDL/
echo "✅ 头文件安装完成"

echo ""
echo "🎉 SnnDL基本库安装完成！"
echo ""
echo "可用组件："
echo "  - SnnDL.SnnPE      (神经网络处理单元)"
echo "  - SnnDL.SpikeSource (脉冲源)"
echo ""
echo "测试命令："
echo "  sst test_basic_snn.py"
echo ""
echo "文档位置："
echo "  - README.md"
echo "  - SnnPE_Port_Interface_Specification.md"
