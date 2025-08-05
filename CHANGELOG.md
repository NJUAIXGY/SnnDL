# Changelog

All notable changes to the SnnDL project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
