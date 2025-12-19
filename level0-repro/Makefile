CXX = g++
CXXFLAGS = -std=c++17 -Wall -g
LDFLAGS = -lze_loader -lpthread

TARGET = repro

all: $(TARGET)

$(TARGET): repro.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	ZE_AFFINITY_MASK=0.0 ./$(TARGET)

.PHONY: all clean run
