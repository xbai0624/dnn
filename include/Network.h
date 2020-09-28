#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "Layer.h"
#include "Matrix.h"

class DataInterface;

class Network 
{
public:
    Network();
    template<typename T>
	Network(T l)
	{
	    __middleAndOutputLayers.push_back(dynamic_cast<Layer*>(l));
	}

    template<typename T, typename... Args>
	Network(T l, Args... pars)
	{
	    Network(pars...);
	}
    ~Network();

    // inits
    void Init();
    void ConstructLayers(TrainingType);

    // training procedures
    void Train();
    void UpdateEpoch();
    void UpdateBatch();
    void ForwardPropagateForBatch();
    void BackwardPropagateForBatch();
    void UpdateWeightsAndBiasForBatch();
    float GetCost();

    // testing procedures
    float GetAccuracy();
    float GetError();

    // work procedures
    std::vector<Matrix> Classify();

private:
    std::vector<Layer*> __middleAndOutputLayers;                  // save all middle layers
    Layer *__inputLayer=nullptr, *__outputLayer=nullptr; // input and output layers
    DataInterface *__dataInterface = nullptr;            // save data interface class

    int __numberOfEpoch = 100; // nuber of epochs
};

#endif
