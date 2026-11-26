CXX ?= c++
CXXFLAGS ?= -std=c++20 -O3 -Wall -Wextra -Werror -pthread -Iinclude

BUILD_DIR := build

.PHONY: all test demo bench clean

all: $(BUILD_DIR)/test_engine $(BUILD_DIR)/main_demo $(BUILD_DIR)/benchmark

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/inference_engine.o: src/inference_engine.cpp include/inference_engine.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/inference_engine.cpp -o $(BUILD_DIR)/inference_engine.o

$(BUILD_DIR)/test_engine: tests/test_engine.cpp $(BUILD_DIR)/inference_engine.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_engine.cpp $(BUILD_DIR)/inference_engine.o -o $(BUILD_DIR)/test_engine

$(BUILD_DIR)/main_demo: src/main_demo.cpp $(BUILD_DIR)/inference_engine.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/main_demo.cpp $(BUILD_DIR)/inference_engine.o -o $(BUILD_DIR)/main_demo

$(BUILD_DIR)/benchmark: src/benchmark.cpp $(BUILD_DIR)/inference_engine.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/benchmark.cpp $(BUILD_DIR)/inference_engine.o -o $(BUILD_DIR)/benchmark

test: $(BUILD_DIR)/test_engine
	./$(BUILD_DIR)/test_engine

demo: $(BUILD_DIR)/main_demo
	./$(BUILD_DIR)/main_demo

bench: $(BUILD_DIR)/benchmark
	./$(BUILD_DIR)/benchmark

clean:
	rm -rf $(BUILD_DIR)
