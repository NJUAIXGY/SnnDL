# Makefile for Neuron Mapping Framework Tests
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -I../include
SRCDIR = ../src
TESTDIR = .

# Source files
CORE_SOURCES = $(SRCDIR)/core/NeuralNetwork.cpp \
               $(SRCDIR)/core/HardwareTopology.cpp \
               $(SRCDIR)/core/MappingSolution.cpp \
               $(SRCDIR)/core/NeuronMapper.cpp

STRATEGY_SOURCES = $(SRCDIR)/strategies/MappingStrategy.cpp \
                   $(SRCDIR)/strategies/RandomMappingStrategy.cpp \
                   $(SRCDIR)/strategies/GreedyMappingStrategy.cpp

OPTIMIZER_SOURCES = $(SRCDIR)/optimizers/MappingOptimizer.cpp \
                    $(SRCDIR)/optimizers/LocalSearchOptimizer.cpp \
                    $(SRCDIR)/optimizers/HillClimbingOptimizer.cpp

EVALUATOR_SOURCES = $(SRCDIR)/evaluators/PerformanceEvaluator.cpp \
                    $(SRCDIR)/evaluators/CommunicationCostEvaluator.cpp \
                    $(SRCDIR)/evaluators/LoadBalanceEvaluator.cpp \
                    $(SRCDIR)/evaluators/ComprehensiveEvaluator.cpp

ALGORITHM_SOURCES = $(SRCDIR)/algorithms/NetworkAnalyzer.cpp

FACTORY_SOURCES = $(SRCDIR)/factories/MapperFactory.cpp

UTIL_SOURCES = $(SRCDIR)/utils/Logger.cpp \
               $(SRCDIR)/utils/MathUtils.cpp

ALL_SOURCES = $(CORE_SOURCES) $(STRATEGY_SOURCES) $(OPTIMIZER_SOURCES) \
              $(EVALUATOR_SOURCES) $(ALGORITHM_SOURCES) $(FACTORY_SOURCES) \
              $(UTIL_SOURCES)

# Test executables
TEST_CORE = test_core
TEST_PERFORMANCE = test_performance

# Default target
all: $(TEST_CORE) $(TEST_PERFORMANCE)

# Core functionality tests
$(TEST_CORE): test_core.cpp $(ALL_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Performance tests
$(TEST_PERFORMANCE): test_performance.cpp $(ALL_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^

# Check compilation of individual components
check-core:
	@echo "Checking core components compilation..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(CORE_SOURCES)
	@echo "Core components compile successfully"

check-strategies:
	@echo "Checking strategies compilation..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(STRATEGY_SOURCES)
	@echo "Strategies compile successfully"

check-optimizers:
	@echo "Checking optimizers compilation..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(OPTIMIZER_SOURCES)
	@echo "Optimizers compile successfully"

check-evaluators:
	@echo "Checking evaluators compilation..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(EVALUATOR_SOURCES)
	@echo "Evaluators compile successfully"

check-all: check-core check-strategies check-optimizers check-evaluators
	@echo "All components compile successfully"

# Run tests
run-core: $(TEST_CORE)
	@echo "=== Running Core Tests ==="
	./$(TEST_CORE)

run-performance: $(TEST_PERFORMANCE)
	@echo "=== Running Performance Tests ==="
	./$(TEST_PERFORMANCE)

run-all: run-core run-performance

# Quick test (compile and run core tests)
test: $(TEST_CORE)
	./$(TEST_CORE)

# Clean build artifacts
clean:
	rm -f $(TEST_CORE) $(TEST_PERFORMANCE) *.o

# Debug build
debug: CXXFLAGS += -DDEBUG -g3
debug: all

# Release build
release: CXXFLAGS += -DNDEBUG -O3
release: all

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Build all test executables"
	@echo "  test-core     - Build and run core functionality tests"
	@echo "  test-performance - Build and run performance tests"
	@echo "  check-all     - Check compilation of all components"
	@echo "  run-all       - Run all tests"
	@echo "  test          - Quick test (core tests only)"
	@echo "  clean         - Remove build artifacts"
	@echo "  debug         - Build with debug flags"
	@echo "  release       - Build with optimization flags"
	@echo "  help          - Show this help message"

.PHONY: all check-core check-strategies check-optimizers check-evaluators check-all
.PHONY: run-core run-performance run-all test clean debug release help