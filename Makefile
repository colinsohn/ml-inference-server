CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Iinclude

all: build/ml-infer-cli build/test_engine

build/ml-infer-cli: src/engine.cpp src/main.cpp include/ml_inference/engine.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/engine.cpp src/main.cpp -o build/ml-infer-cli

build/test_engine: src/engine.cpp tests/test_engine.cpp include/ml_inference/engine.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/engine.cpp tests/test_engine.cpp -o build/test_engine

test: all
	python3 -B scripts/build_and_test.py

clean:
	rm -rf build artifacts/*.bin
