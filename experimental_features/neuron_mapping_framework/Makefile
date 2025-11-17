CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g
INCLUDES = -I include
SRCDIR = src
EXAMPLESDIR = examples
TESTSDIR = tests

# Source files
CORE_SOURCES = $(wildcard $(SRCDIR)/core/*.cpp)
ALGORITHMS_SOURCES = $(wildcard $(SRCDIR)/algorithms/*.cpp)
EVALUATORS_SOURCES = $(wildcard $(SRCDIR)/evaluators/*.cpp)
OPTIMIZERS_SOURCES = $(wildcard $(SRCDIR)/optimizers/*.cpp)
STRATEGIES_SOURCES = $(wildcard $(SRCDIR)/strategies/*.cpp)
ROUTING_SOURCES = $(wildcard $(SRCDIR)/routing/*.cpp)
UTILS_SOURCES = $(wildcard $(SRCDIR)/utils/*.cpp)

ALL_SOURCES = $(CORE_SOURCES) $(ALGORITHMS_SOURCES) $(EVALUATORS_SOURCES) \
              $(OPTIMIZERS_SOURCES) $(STRATEGIES_SOURCES) $(ROUTING_SOURCES) $(UTILS_SOURCES)

# Object files
OBJECTS = $(ALL_SOURCES:.cpp=.o)

# Executables
WORKING_DEMO = $(EXAMPLESDIR)/working_demo
WORKING_TEST = $(TESTSDIR)/test_working_integration
COMMUNITY_TEST = $(TESTSDIR)/test_community_detection
BENCHMARK_TEST = $(EXAMPLESDIR)/benchmark_test
ROUTING_DEMO = $(EXAMPLESDIR)/routing_demo
ROUTING_TEST = $(TESTSDIR)/test_routing_system
ROUTING_EXPORT = tools/routing_exporter
EXPORT_TEST = $(TESTSDIR)/test_export_artifacts
LARGE_EXPERIMENT = tools/routing_experiment_large

# Default target
all: $(WORKING_DEMO) $(WORKING_TEST) $(COMMUNITY_TEST) $(BENCHMARK_TEST) $(ROUTING_DEMO) $(ROUTING_TEST) $(ROUTING_EXPORT) $(EXPORT_TEST) $(LARGE_EXPERIMENT)

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build working demo
$(WORKING_DEMO): $(OBJECTS) $(EXAMPLESDIR)/working_demo.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(EXAMPLESDIR)/working_demo.o -o $@

# Build working integration test
$(WORKING_TEST): $(OBJECTS) $(TESTSDIR)/test_working_integration.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TESTSDIR)/test_working_integration.o -o $@

# Build community detection test
$(COMMUNITY_TEST): $(OBJECTS) $(TESTSDIR)/test_community_detection.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TESTSDIR)/test_community_detection.o -o $@

# Build benchmark test
$(BENCHMARK_TEST): $(OBJECTS) $(EXAMPLESDIR)/benchmark_test.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(EXAMPLESDIR)/benchmark_test.o -o $@

# Build routing demo
$(ROUTING_DEMO): $(OBJECTS) $(EXAMPLESDIR)/routing_demo.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(EXAMPLESDIR)/routing_demo.o -o $@

# Build routing test
$(ROUTING_TEST): $(OBJECTS) $(TESTSDIR)/test_routing_system.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TESTSDIR)/test_routing_system.o -o $@

# Build routing exporter tool
$(ROUTING_EXPORT): $(OBJECTS) tools/routing_exporter.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) tools/routing_exporter.o -o $@

$(LARGE_EXPERIMENT): $(OBJECTS) tools/routing_experiment_large.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) tools/routing_experiment_large.o -o $@

# Build example object files
$(EXAMPLESDIR)/working_demo.o: $(EXAMPLESDIR)/working_demo.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(EXAMPLESDIR)/benchmark_test.o: $(EXAMPLESDIR)/benchmark_test.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(EXAMPLESDIR)/routing_demo.o: $(EXAMPLESDIR)/routing_demo.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build test object files
$(TESTSDIR)/test_working_integration.o: $(TESTSDIR)/test_working_integration.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TESTSDIR)/test_community_detection.o: $(TESTSDIR)/test_community_detection.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(TESTSDIR)/test_routing_system.o: $(TESTSDIR)/test_routing_system.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(EXPORT_TEST): $(OBJECTS) $(TESTSDIR)/test_export_artifacts.o
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TESTSDIR)/test_export_artifacts.o -o $@

$(TESTSDIR)/test_export_artifacts.o: $(TESTSDIR)/test_export_artifacts.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tools/routing_exporter.o: tools/routing_exporter.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

tools/routing_experiment_large.o: tools/routing_experiment_large.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test compilation of individual source files
test-compile:
	@echo "Testing compilation of all source files..."
	@for file in $(ALL_SOURCES); do \
		echo "Compiling $$file..."; \
		$(CXX) $(CXXFLAGS) $(INCLUDES) -c $$file -o /tmp/temp.o || exit 1; \
	done
	@echo "All source files compile successfully!"

# Run the working demo
run-demo: $(WORKING_DEMO)
	@echo "Running working demo..."
	@./$(WORKING_DEMO)

# Run the working integration test
run-test: $(WORKING_TEST)
	@echo "Running working integration test..."
	@./$(WORKING_TEST)

# Run community detection test
run-community-test: $(COMMUNITY_TEST)
	@echo "Running community detection test..."
	@./$(COMMUNITY_TEST)

# Run benchmark test
run-benchmark: $(BENCHMARK_TEST)
	@echo "Running benchmark test..."
	@./$(BENCHMARK_TEST)

# Run benchmark test with custom parameters
run-benchmark-large: $(BENCHMARK_TEST)
	@echo "Running large-scale benchmark test (512 neurons, 32 PEs)..."
	@./$(BENCHMARK_TEST) 512 32

# Run routing demo
run-routing-demo: $(ROUTING_DEMO)
	@echo "Running routing system demo..."
	@./$(ROUTING_DEMO)

# Run routing tests
run-routing-test: $(ROUTING_TEST)
	@echo "Running routing system tests..."
	@./$(ROUTING_TEST)

# Run routing exporter (writes to ./export)
run-export: $(ROUTING_EXPORT)
	@echo "Exporting mapping and routing artifacts to ./export ..."
	@./$(ROUTING_EXPORT)

# Run all tests
run-all: run-demo run-test run-community-test run-benchmark run-routing-demo run-routing-test

# Clean build files
clean:
	@echo "Cleaning build files..."
	@find . -name "*.o" -delete
	@rm -f $(WORKING_DEMO) $(WORKING_TEST) $(COMMUNITY_TEST) $(BENCHMARK_TEST) $(ROUTING_DEMO) $(ROUTING_TEST)
	@echo "Clean completed."

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build all executables"
	@echo "  test-compile     - Test compilation of all source files"
	@echo "  run-demo         - Build and run the working demo"
	@echo "  run-test         - Build and run the working integration test"
	@echo "  run-community-test - Build and run community detection test"
	@echo "  run-benchmark    - Build and run benchmark test"
	@echo "  run-routing-demo - Build and run routing system demo"
	@echo "  run-routing-test - Build and run routing system tests"
		@echo "  run-export       - Build and run routing exporter (writes JSON/CSV to ./export)"
		@echo "  run-export-test  - Build and run exportArtifacts API test (writes to ./export_test)"
		@echo "  run-large        - Build and run large-scale experiment (summary to ./export_large)"

run-export-test: $(EXPORT_TEST)
	@echo "Running exportArtifacts API test..."
	@./$(EXPORT_TEST)

run-large: $(LARGE_EXPERIMENT)
		@echo "Running large-scale routing experiment (this may take long)..."
		@./$(LARGE_EXPERIMENT)

.PHONY: all test-compile run-demo run-test run-community-test run-benchmark run-routing-demo run-routing-test run-all clean help
