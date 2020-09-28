/*
 * neuron class
 *     the weights are defined as a matrix, so we can unify cnn neurons and MLP neurons
 *         --MLP neurons weight matrix dimension is (1, n); 
 *         --CNN neuron weight matrix dimension is (m, n) --- each CNN neuron will have one independent copy of its corresponding kernel
 *
 *     cnn neuron and mlp neuron all only have one bias;
 */
#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cstddef>

#include "Layer.h"

class Neuron 
{
public:
    Neuron();
    ~Neuron();

    void PassWeightPointer(Matrix *);
    void PassBiasPointer(Matrix *);

    // 
    void ForwardPropagateForSample(int sample_index);
    void BackwardPropagateForSample(int sample_index);

    bool IsActive(); // for dropout
    void Disable();
    void Enable();

    void Reset(); // reset for next training (batch level)

    // actuation function
    double __sigmoid(double);   // sigmoid
    double __softmax(Matrix &); // softmax
    double __tanh(double);      // tanh(x)
    double __relu(double);      // rectified linear unit

    // setters
    void SetLayer(Layer*); // layer information
    //void SetPreviousLayer(Layer*);
    //void SetNextLayer(Layer *);
    void SetCoord(size_t _i=0, size_t _j=0, size_t _k=0); // set neuron(pixel) coordinates in current layer
    void SetCoordI(size_t);
    void SetCoordJ(size_t);
    void SetCoordK(size_t);
    void SetCoord(NeuronCoord);
    void SetActuationFuncType(ActuationFuncType t);
    ActuationFuncType GetActuationFuncType();

    // update matrix a; 'a' computation is independent of layer type, 
    // no need to design helper functions for different type layers
    void UpdateA(int sample_id); 
    // update derivative of a over z; sigma^prime
    void UpdateSigmaPrime(int sample_id); 

    void UpdateDelta(int sample_index); // update matrix delta
    void UpdateDeltaCNN(int sample_index);
    void UpdateDeltaFC(int sample_index);
    void UpdateDeltaOutputLayer(int sample_index);
    void UpdateDeltaPooling(int sample_index);

    void UpdateZ(int sample_id); // update matrix z
    void UpdateZCNN(int sample_id);
    void UpdateZFC(int sample_id);
    void UpdateZPooling(int sample_id);

    void ClearPreviousBatch();

    // helper for batch normalization
    void ReplaceZ(int sample_id, double val);

    // getters 
    std::vector<double> & GetAVector();
    std::vector<double> & GetDeltaVector();
    std::vector<double> & GetZVector();
    std::vector<double> & GetSigmaPrimeVector();
    NeuronCoord GetCoord();

    // 
    void Print();

private:
    static long __neuron_Count;
    long __neuron_id;
    Matrix *__w;  // weights
    Matrix *__b;  // bias : should be a 1 element matrix

    // for batch training, after each batch, these vectors will be cleared for next batch
    std::vector<double> __a; // the length is dependent on training batch size
    std::vector<double> __delta;
    std::vector<double> __z;
    std::vector<double> __sigmaPrime; // derivative of Z

    // the layer that this neurons belongs to
    Layer *__layer = nullptr;
    //Layer *__previousLayer = nullptr;
    //Layer *__nextLayer = nullptr;

    // for dropout algorithm, dropout algorithm will disable
    // some neurons in a layer
    bool __active_status = true; 
    // actuation function type used in this neuron
    ActuationFuncType __funcType = ActuationFuncType::Sigmoid;

    NeuronCoord __coord; // active coordinates
};

#endif
