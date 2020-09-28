g++ -std=c++11 -o matrix_multiply matrix_multiply.cpp 
g++ -std=c++11 -o sigmoid_func sigmoid_func.cpp
g++ -std=c++11 -o softmax_func softmax_func.cpp
g++ -std=c++11 -o differential differential.cpp
g++ -std=c++11 -o plot_loss plot_loss.cpp -I${ROOTSYS}/include $(root-config --glibs)
