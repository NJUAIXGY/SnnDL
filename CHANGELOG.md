# Changelog

All notable changes to the SnnDL project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-08-05

### Added
- ✅ **完全功能的基础神经网络系统**
- 🚀 **Handler2迁移**: 现代SST事件处理机制
- 🔧 **序列化修复**: SpikeEvent支持网络传输
- 📁 **权重加载系统**: 从文件正确加载神经连接
- ⏰ **时间同步机制**: SpikeSource和SnnPE时钟协调
- 🌐 **节点间通信**: 多SnnPE组件脉冲传播
- 📊 **统计监控**: 完整的性能指标收集
- 🧠 **LIF神经元模型**: 正确的发放和传播机制
- 🔗 **传统Link模式**: 完全正常工作的链接系统
- 🧪 **2节点测试**: 验证扩展SpikeEvent传播

### Fixed
- 🐛 **权重文件加载**: 修复空字符串导致的加载失败
- ⏱️ **时间单位转换**: 修正SpikeSource时钟周期到纳秒的转换
- 🔧 **程序挂起问题**: 解决2节点配置中的执行停滞
- 📝 **调试输出**: 添加完整的执行流程跟踪

### Changed
- 🔄 **Clock::Handler** → **Clock::Handler2** 迁移
- 📦 **ImplementSerializable** 宏用于现代序列化
- 🎯 **神经元参数优化**: 降低阈值，提高权重，减少不应期
- 📈 **增强调试**: 详细的printf调试信息

### Technical Details
- **验证的数据流**: SpikeSource(3) → SnnPE0(3→5) → SnnPE1(5→0)
- **神经元活动**: 5个神经元发放，5次突触操作
- **库状态**: libSnnDL.so正确编译和安装
- **统计完整性**: CSV格式性能指标导出

## [1.0.0] - 2025-08-05

### Added
- 🎉 Initial release of SnnDL framework
- ⚡ Core SnnPE component for spiking neural network processing
- 📡 SpikeSource component for data injection
- 🌐 Network-aware SpikeEvent with routing capabilities
- 🔌 Miranda-style SubComponent interface architecture
- 🛜 SnnNIC network interface controller
- 📊 Built-in statistics and monitoring
- 🧪 Comprehensive test suite
- 📚 Complete documentation and examples

### Features
- **High-Performance Simulation**: Leverages SST framework for parallel execution
- **Distributed Communication**: Supports multi-node spiking neural networks
- **Pluggable Network Interfaces**: Modular network adapter architecture
- **Multiple Data Formats**: TEXT, NMNIST-AER, SHD-HDF5 support
- **Scalable Architecture**: From single-core to cluster deployments

### Components
- `SnnPE`: Spiking Neural Network Processing Element
- `SpikeSource`: Neural spike data source
- `SnnInterface`: Abstract network interface base class
- `SnnNIC`: SimpleNetwork adapter implementation
- `SpikeEvent`: Extended event class with network routing

### Technical Specifications
- SST Core: >= 15.0.0
- C++17 standard compliance
- Linux support (Ubuntu 20.04+ recommended)
- Optional MPI for distributed simulation

### Documentation
- Comprehensive README with installation guide
- API documentation and examples
- Performance benchmarks and analysis
- Contributing guidelines

### Testing
- Unit tests for all components
- Integration tests for network communication
- Performance benchmarking suite
- Continuous integration setup

---

## Future Releases

### [1.1.0] - Planned
- 🔧 Enhanced SubComponent interface activation
- 🚀 Performance optimizations for large-scale networks
- 📈 Advanced statistics and visualization tools
- 🌍 Multi-topology network support

### [1.2.0] - Planned
- 🧠 Learning algorithms integration
- 🔄 Dynamic weight adaptation
- 💾 Checkpoint/restart functionality
- 🔍 Debugging and profiling tools
