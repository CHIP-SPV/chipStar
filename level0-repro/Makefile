CXX = g++
CXXFLAGS = -std=c++17 -Wall -g
LDFLAGS = -lze_loader -lpthread

all: repro repro_min

repro: repro.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

repro_min: repro_min.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f repro repro_min

.PHONY: all clean
