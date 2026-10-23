CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -flto -Wall -Wextra -pthread
LDFLAGS  ?= -flto -pthread

SRC := $(wildcard src/*.cpp)
OBJ := $(SRC:src/%.cpp=build/%.o)

all: dnn

dnn: $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

build/%.o: src/%.cpp $(wildcard src/*.h)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf build dnn

.PHONY: all clean
