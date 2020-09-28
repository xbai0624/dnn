#!/bin/bash
#g++ -std=c++11 -I${ROOTSYS}/include $(root-config --glibs) -o main sum_matrix.cpp
g++ -std=c++11 -I${ROOTSYS}/include $(root-config --glibs) -o main histo_normalization_result.cpp

