# Neuron Mapping Framework Tests

This directory contains comprehensive test suites for the Neuron Mapping Framework, designed to validate core functionality and measure performance characteristics.

## Test Structure

### Core Tests (`test_core.cpp`)
Validates the fundamental functionality of all major components:

- **NeuralNetwork Tests**: Creation, modification, and querying of neural networks
- **HardwareTopology Tests**: Topology creation, distance calculations, and PE management
- **MappingSolution Tests**: Neuron assignment, removal, and solution management
- **NeuronMapper Tests**: End-to-end mapping workflows and component integration
- **Factory Pattern Tests**: Mapper creation and configuration validation
- **Integration Tests**: Complete workflow testing with realistic scenarios

### Performance Tests (`test_performance.cpp`)
Measures performance characteristics across different scenarios:

- **Network Scaling**: Performance impact of increasing network size
- **Topology Performance**: Efficiency of different hardware topologies
- **Strategy Comparison**: Performance trade-offs between mapping strategies
- **Memory Performance**: Memory allocation patterns and efficiency
- **Concurrent Performance**: Multi-network mapping scenarios

## Building and Running Tests

### Prerequisites
- C++17 compatible compiler (g++ recommended)
- Standard C++ libraries
- Make utility

### Build Commands

```bash
# Build all tests
make all

# Build specific tests
make test_core
make test_performance

# Quick compilation check
make check-all
```

### Running Tests

```bash
# Run core functionality tests
make run-core

# Run performance tests  
make run-performance

# Run all tests
make run-all

# Quick test (core only)
make test
```

### Build Variants

```bash
# Debug build (with debug symbols and assertions)
make debug

# Release build (optimized)
make release

# Clean build artifacts
make clean
```

## Test Output

### Core Tests
The core tests use a simple test framework that reports:
- Individual test results (`[PASS]` or `[FAIL]`)
- Test summary with pass/fail counts and success rate
- Detailed failure information when tests fail

Example output:
```
=== Testing NeuralNetwork ===
[PASS] Initial neuron count should be 0
[PASS] Add neuron 0
[PASS] Should have 5 neurons after adding
...

=== Test Summary ===
Total tests: 45
Passed: 43  
Failed: 2
Success rate: 95.6%
```

### Performance Tests
Performance tests output timing information and metrics:
- Execution times in milliseconds
- Scalability characteristics
- Memory usage patterns
- Performance comparisons between strategies

Example output:
```
=== Network Scaling Performance Test ===
    Neurons    Connections   Map Time (ms)   Eval Time (ms)
    --------------------------------------------------------
          10             15            2.45            0.15
          25             37            5.12            0.28
         100            150           18.73            1.45
```

## Test Coverage

The test suite covers:

### Functional Coverage
- ✅ Data structure creation and manipulation
- ✅ Algorithm correctness verification
- ✅ Error handling and edge cases
- ✅ Interface compliance
- ✅ Integration between components

### Performance Coverage  
- ✅ Scalability with network size
- ✅ Efficiency of different topologies
- ✅ Strategy performance comparison
- ✅ Memory allocation patterns
- ✅ Concurrent operation simulation

### Edge Cases
- ✅ Empty networks and topologies
- ✅ Single-node scenarios
- ✅ Maximum capacity constraints
- ✅ Invalid configuration handling
- ✅ Resource exhaustion scenarios

## Interpreting Results

### Core Test Results
- **100% pass rate**: All components working correctly
- **95%+ pass rate**: Acceptable with minor issues
- **<95% pass rate**: Indicates significant problems requiring investigation

### Performance Benchmarks
- **Mapping time**: Should scale sub-quadratically with network size
- **Evaluation time**: Should scale linearly with network complexity  
- **Memory usage**: Should be proportional to network size without leaks
- **Strategy comparison**: Trade-offs between speed and solution quality

## Troubleshooting

### Common Build Issues

1. **Compilation errors**: Check C++17 support and include paths
2. **Linking errors**: Verify all source files are included in Makefile
3. **Missing headers**: Ensure include directory structure is correct

### Common Test Failures

1. **Network creation failures**: Check neuron ID uniqueness and connection validity
2. **Topology distance errors**: Verify topology initialization and coordinate calculations
3. **Mapping failures**: Check PE capacity constraints and assignment logic
4. **Performance variations**: Results may vary based on system load and hardware

### Debugging Failed Tests

1. Enable debug build: `make debug`
2. Run specific failing tests with detailed output
3. Check log messages for component-level debugging
4. Use debugger to step through failing scenarios

## Extending Tests

### Adding New Test Cases

1. Add test functions to appropriate test file
2. Call new functions from main()
3. Use TestFramework assertions for validation
4. Update Makefile if new source files required

### Adding Performance Tests

1. Create test scenarios in `test_performance.cpp`
2. Use PerformanceTimer class for timing
3. Output results in tabular format for easy analysis
4. Include both average and worst-case measurements

## Continuous Integration

These tests are designed to be run in CI/CD environments:

```bash
# CI-friendly test execution
make clean && make check-all && make test
```

Exit codes:
- `0`: All tests passed
- `1`: Some tests failed or build errors occurred

## Best Practices

1. **Run tests before commits**: Ensure no regressions
2. **Add tests for new features**: Maintain coverage
3. **Profile performance regularly**: Catch performance regressions
4. **Test edge cases**: Verify robustness
5. **Update tests with API changes**: Keep tests synchronized