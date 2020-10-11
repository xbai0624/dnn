#include "Layer.h"

#include <iostream>
#include <iomanip>

std::ostream & operator<<(std::ostream &os, const LayerType &t)
{
    if(t == LayerType::fullyConnected)
        os<<"____layer_type=fully_connected____";
    else if(t == LayerType::cnn)
        os<<"____layer_type=cnn____";
    else if(t == LayerType::pooling)
        os<<"____layer_type=pooling____";
    else if(t == LayerType::input)
        os<<"____layer_type=input____";
    else if(t == LayerType::output)
        os<<"____layer_type=output____";
    else if(t == LayerType::Undefined)
        os<<"____layer_type=undefined____";
    return os;
}

std::ostream & operator<<(std::ostream &os, const ActuationFuncType & t)
{
    if(t == ActuationFuncType::Sigmoid)
        os<<"Sigmoid_Actuation_Func";
    else if(t == ActuationFuncType::SoftMax)
        os<<"Softmax_Actuation_Func";
    else if(t == ActuationFuncType::Tanh)
        os<<"Tanh_Actuation_Func";
    else if(t == ActuationFuncType::Relu)
        os<<"Relu_Actuation_Func";
    else 
        os<<"Undefined_Actuation_Func";
    return os;
}

std::ostream & operator<<(std::ostream &os, const LayerDimension &dim)
{
    if(dim == LayerDimension::_1D)
        os<<"__LayerDimension=_1D__";
    else if(dim == LayerDimension::_2D)
        os<<"__LayerDimension=_2D__";
    else if(dim == LayerDimension::Undefined)
        os<<"__LayerDimension=Undefined__";
    return os;
}

std::ostream & operator<<(std::ostream &os, const Filter2D &t)
{
    for(size_t i=0;i<t.__filter.size();i++)
    {
        for(size_t j=0;j<t.__filter[i].size();j++)
        {
            os<<std::setfill(' ')<<std::setw(4)<<t.__filter[i][j];
        }
        os<<std::endl;
    }
    return os;
}

std::ostream & operator<<(std::ostream& os, const NeuronCoord &c)
{
    os<<std::setfill(' ')<<std::setw(4)<<c.i
        <<std::setfill(' ')<<std::setw(4)<<c.j
        <<std::setfill(' ')<<std::setw(4)<<c.k
        <<std::endl;
    return os;
}

std::ostream & operator<<(std::ostream &os, const Images & images)
{
    os<<"images from all kernels during one training sample:"<<std::endl;
    for(size_t i=0;i<images.OutputImageFromKernel.size();i++)
    {
        os<<"kernel: "<<i<<std::endl;
        Matrix m = (images.OutputImageFromKernel)[i];
        os<<m<<std::endl;
    }
    return os;
}



int Layer::__layerCount = 0;

Layer::Layer()
{
    // place holder
    __layerID = __layerCount;
    __layerCount++;
}

Layer::~Layer()
{
    // place holder
}
