CXX      = g++
CXXFLAGS = -std=c++17 -O2 -fopenmp

# ---- QuantLib configuration ----
# Option A (Linux/macOS): use quantlib-config if available
QL_CFLAGS := $(shell quantlib-config --cflags 2>/dev/null)
QL_LIBS   := $(shell quantlib-config --libs   2>/dev/null)

# Option B (Windows / manual install): uncomment and set paths
# QL_INCLUDE    = C:/vcpkg/installed/x64-windows/include
# BOOST_INCLUDE = C:/vcpkg/installed/x64-windows/include
# QL_LIBDIR     = C:/vcpkg/installed/x64-windows/lib
# QL_CFLAGS     = -I$(QL_INCLUDE) -I$(BOOST_INCLUDE)
# QL_LIBS       = -L$(QL_LIBDIR) -lQuantLib -lboost_system

ifeq ($(strip $(QL_CFLAGS)),)
  $(warning quantlib-config not found — set QL_CFLAGS / QL_LIBS manually in this Makefile.)
endif

# ---- Targets ----
TARGET = runner2

all: $(TARGET)

$(TARGET): runner2.cpp
	$(CXX) $(CXXFLAGS) $(QL_CFLAGS) -o $@ $< $(QL_LIBS)

clean:
	rm -f $(TARGET) $(TARGET).exe

.PHONY: all clean
