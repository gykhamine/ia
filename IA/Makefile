# ===========================================================================
# IA/cpp — Makefile pour _ia_core.so (sans CMake, sans pybind11)
#
# Dep: g++, make, numpy (interface Python uniquement)
# AUCUNE dep C++ externe
#
# Usage:
#   make            # compile
#   make clean      # nettoie
#   make test       # compile + test
#   make info       # infos
# ===========================================================================

PYTHON       ?= python3
SUFFIX       ?= $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

CXX         ?= g++
CXX_FLAGS   := -std=c++17 -O3 -march=native -ffast-math -fPIC -fopenmp
INCLUDES    := -Icpp

SOURCES     := cpp/c_api.cpp cpp/engine.cpp cpp/autograd.cpp
TARGET      := cpp/_ia_core$(SUFFIX)

LDFLAGS     := -shared -pthread -fopenmp

.PHONY: all clean test info

default: all

all: $(TARGET)

$(TARGET): $(SOURCES) cpp/engine.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^
	@echo ""
	@echo "Module compile : $(TARGET) (C++ pur, ni CMake ni pybind11)"

test: $(TARGET)
	$(PYTHON) -c "import sys;sys.path.insert(0,'.');from cpp import get_core;C=get_core();import numpy as np;a=np.array([[1.,2.],[3.,4.]]);b=np.array([[5.,6.],[7.,8.]]);print('matmul:',C.matmul(a,b));print('relu:',C.relu(np.array([-1.,0.5,2.])));print('OK')"

clean:
	rm -f cpp/_ia_core*.so
	@echo "Nettoye."

info:
	@echo "CXX       : $(CXX)"
	@echo "PYTHON    : $(PYTHON)"
	@echo "SUFFIX    : $(SUFFIX)"
	@echo "SOURCES   : $(SOURCES)"
	@echo "TARGET    : $(TARGET)"
