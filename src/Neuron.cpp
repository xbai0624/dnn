#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cassert>

#include "Layer.h"
#include "Neuron.h"
#include "Matrix.h"
#include "DataInterface.h"

using namespace std;

long Neuron::__neuron_Count = 0;

Neuron::Neuron()
{
    // place holder
    __neuron_id = __neuron_Count;
    __neuron_Count++;
}

Neuron::~Neuron()
{
    // place holder
}

void Neuron::PassWeightPointer(Matrix *m)
{
    // pass weight matrix pointer
    // matrix dimension is set in Layer class
    __w = m;
}

void Neuron::PassBiasPointer(Matrix *_b)
{
    // pass bias pointer
    __b = _b;
}

void Neuron::ForwardPropagateForSample(int sample_index)
{
    UpdateZ(sample_index);
    UpdateA(sample_index);
    UpdateSigmaPrime(sample_index);
}

void Neuron::BackwardPropagateForSample(int sample_index)
{
    UpdateDelta(sample_index);
}


double Neuron::__sigmoid(double z)
{
    return 1/(1 + exp(-z));
}


double Neuron::__softmax(Matrix &m_z)
{
    if(__layer->GetType() != LayerType::output)
    {
        // soft max is usually used in output layer, it is rare in hidden layer
        // print a warning if the neuron detected layertype not = output
        std::cout<<__func__<<" Warning: using softmax in hidden layers? please make sure."
            <<std::endl;
    }
    auto dim = m_z.Dimension();
    assert(dim.second == 1);

    size_t i = __coord.i;
    assert(dim.first > i);

    double sum = 0;
    for(size_t k = 0;k<dim.first;k++)
    {
        sum += exp((double)m_z[k][0]);
    }

    return exp(m_z[i][0])/sum;
}



double Neuron::__tanh(double z)
{
    double _t = exp(2.0*z);
    return (_t - 1.)/(_t +1.);
}

double Neuron::__relu(double z)
{
    // return max(0, z);
    if(z < 0.) return 0;
    return z;
}

void Neuron::Disable()
{
    __active_status = false;
}

void Neuron::Enable()
{
    __active_status = true;
}

bool Neuron::IsActive()
{
    return __active_status;
}

void Neuron::Reset()
{
    if(__layer == nullptr)
    {
        std::cout<<"Error: Neuron::Reset(): the layer info has not been set for this neuron."
            <<endl;
        exit(0);
    }
    int batch_size = __layer->GetBatchSize();
    // after updateing weights and biase, 
    // namely after one training (one sample or one batch depends on user)
    // reset neurons for next computation
    __a.clear();
    __a.resize(batch_size); // reset a
    __delta.clear();
    __delta.resize(batch_size); // reset delta
    __z.clear();
    __z.resize(batch_size); // reset z
    __sigmaPrime.clear();
    __sigmaPrime.resize(batch_size);

    //__wGradient.clear(); // reset weight gradient
    //__bGradient.clear(); // reset bias gradient
}

void Neuron::SetLayer(Layer* l)
{
    // set layer that this neuron currently belongs to
    __layer = l;
}

//void Neuron::SetPreviousLayer(Layer* l)
//{
// set neuron's prevous layer
//__previousLayer = l;
//}

//void Neuron::SetNextLayer(Layer* l)
//{
// set neurons's next layer
//__nextLayer = l;
//}

void Neuron::SetActuationFuncType(ActuationFuncType t)
{
    // set neurons's actuation function type
    __funcType = t;
}


ActuationFuncType Neuron::GetActuationFuncType()
{
    // get neurons's actuation function type
    return __funcType;
}


void Neuron::UpdateZ(int sample_index)
{
    // update z for current layer
    if(__layer->GetType() == LayerType::fullyConnected || __layer->GetType() == LayerType::output)
    {
        //std::cout<<__func__<<" fully connected layer update z"<<std::endl;
        UpdateZFC(sample_index);
    } 
    else if(__layer->GetType() == LayerType::cnn)
    {
        //std::cout<<__func__<<" cnn layer update z"<<std::endl;
        UpdateZCNN(sample_index);
    } 
    else if(__layer->GetType() == LayerType::pooling)
    {
        //std::cout<<__func__<<" pooling layer update z"<<std::endl;
        UpdateZPooling(sample_index);
    } 
    else 
    {
        std::cout<<"Error: unsupported layer type."<<std::endl;
        exit(0);
    }
}

void Neuron::UpdateZFC(int sample_index)
{
    // for fully connected layer, matrix reform are also done in here
    Layer* __previousLayer = __layer->GetPrevLayer();
    std::vector<Images>& _t = __previousLayer -> GetImagesActiveA(); // this line with '&' is 5 times faster than the following line without '&'

    //cout<<"batch size: "<<_t.size()<<endl;
    //cout<<"kernel number: "<<_t[0].GetNumberOfKernels()<<endl;
    //cout<<" image dimension: "<<_t[0].OutputImageFromKernel[0].Dimension()<<endl;
    if(_t.size() < 1 || _t.size() < (size_t)sample_index) 
    {
        std::cout<<"Error: previous layer has not 'A' image."<<std::endl;
        exit(0);
    }
    //Images &images = _t[sample_index]; // get images for current sample


    double z_for_current_neuron = 0;
    // get images for current sample
    Images images;
    if(__previousLayer->GetType() != LayerType::fullyConnected/* && __previousLayer->GetType() != LayerType::input*/ )
    {
        // need vectorization 2D->1D
        // input layer vectorization is already done in DataInterface class
        images = _t[sample_index].Vectorization(); // get images for current sample
    }
    else
    {
        // previous layer is fc, no need to do anything
        images = _t[sample_index];
    }

    if(images.OutputImageFromKernel.size() != 1) 
    {
        std::cout<<"Eroor: layer type not match, expecting FC layer, FC layer should only have 1 kernel."<<std::endl;
        exit(0);
    }

    Matrix &image = images.OutputImageFromKernel[0]; // FC layer has only one "kernel" (equivalent kernel)
    //cout<<"neuron id: "<<__neuron_id<<", image A debug:"<<endl;
    //cout<<image<<endl;


    //cout<<__func__<<" CHECKING HERE!!!! weight matrix dimension: "<<(*__w).Dimension()<<endl;
    //cout<<(*__w)<<endl;
    Matrix res = (*__w) * image;
    //cout<<"res matrix: "<<endl;
    //cout<<res<<endl;
    auto dim = res.Dimension();
    if(dim.first != 1 || dim.second != 1) 
    {
        std::cout<<"Error: wrong dimension, expecting 1D matrix."<<std::endl;
    }
    z_for_current_neuron = res[0][0];

    z_for_current_neuron = z_for_current_neuron + (*__b)[0][0];

    __z[sample_index] = z_for_current_neuron;
    //cout<<__func__<<" finished."<<endl;
}

void Neuron::UpdateZCNN(int sample_index)
{
    // cnn layer
    // every single output image needs input from all input images
    Layer *__previousLayer = __layer->GetPrevLayer();
    auto & inputImage = __previousLayer->GetImagesActiveA(); // no tensorization needed; b/c "fc->cnn" type connection is not used at the moment

    auto w_dim = __w->Dimension();
    int stride = __layer->GetCNNStride();

    size_t i_start = __coord.i*stride;
    size_t j_start = __coord.j*stride;
    size_t i_end = i_start + w_dim.first;
    size_t j_end = j_start + w_dim.second;

    //auto &current_sample_image = (inputImage.back()).OutputImageFromKernel;
    auto &current_sample_image = (inputImage[sample_index]).OutputImageFromKernel;
    auto image_dim = current_sample_image[0].Dimension();
    if(i_end > image_dim.first || j_end > image_dim.second)
    {
        std::cout<<"Error: cnn z update: matrix dimension not match, probably due to wrong padding."<<std::endl;
        exit(0);
    }

    // compute z
    double res = 0;
    for(auto &m: current_sample_image)
    {
        for(size_t i=0;i<w_dim.first;i++){
            for(size_t j=0;j<w_dim.second;j++){
                res += m[i+i_start][j+j_start] * (*__w)[i][j];
            }
        }
    }
    double z = res + (*__b)[0][0];
    __z[sample_index] = z;
}

void Neuron::UpdateZPooling(int sample_index)
{
    // pooling layer
    // should be with cnn layer, just kernel matrix all elements=1, bias = 0;
    Layer* __previousLayer = __layer->GetPrevLayer();
    auto & inputImage = __previousLayer->GetImagesActiveA(); // no tensorization needed, b/c "fc->pooling" type connection is not used at the moment
    //if(inputImage.back().OutputImageFromKernel.size() < __coord.k)
    if(inputImage[sample_index].OutputImageFromKernel.size() < __coord.k)
    {
        // output image for current sample
        // pooling layer is different with cnn layer
        // in pooling layer, kernel and 'A' images has a 1-to-1 mapping relationship
        // for pooling layer, number of kernels (previous layer)  = number of kernels (current layer)
        std::cout<<"Error: pooling operation matrix dimension not match"<<std::endl;
        exit(0);
    }
    //Images image = inputImage.back(); // images for current training sample
    Images & image = inputImage[sample_index]; // images for current training sample
    std::vector<Matrix> & images = image.OutputImageFromKernel;
    //for(auto &i: images)cout<<i<<endl;
    Matrix & kernel_image = images[__coord.k];
    //std::cout<<kernel_image<<std::endl;

    auto dim = __w->Dimension();
    size_t i_size = dim.first, j_size = dim.second;

    size_t i_start = __coord.i * i_size;
    size_t j_start = __coord.j * j_size;

    // get pooling method
    PoolingMethod __poolingMethod = __layer->GetPoolingMethod();

    double z;
    if(__poolingMethod == PoolingMethod::Max) 
    {
        z = kernel_image.MaxInSectionWithPadding(i_start, i_start + i_size, j_start, j_start+j_size);
    } 
    else if(__poolingMethod == PoolingMethod::Average) 
    {
        z = kernel_image.AverageInSectionWithPadding(i_start, i_start + i_size, j_start, j_start+j_size);
    }
    else 
    {
        std::cout<<"Error: unspported pooling method, only max and average supported."<<std::endl;
        exit(0);
    }

    //__z.push_back(z);
    __z[sample_index] = z;
}

void Neuron::UpdateA(int sample_index)
{
    // update a for current training sample
    double v = __z[sample_index];
    //cout<<"debug: sample_id: "<<sample_index<<": z = "<<v<<endl;
    double a = -100.; // theorectially, a>=-1
    if(__funcType == ActuationFuncType::Sigmoid)
    {
        a = __sigmoid(v);
        //cout<<"sigmoid"<<endl;
    }
    else if(__funcType == ActuationFuncType::SoftMax)
    {
        auto & images  = __layer->GetImagesActiveZ();
        assert(images[sample_index].OutputImageFromKernel.size() == 1); // make sure it is 1D layer
        Matrix & m_z = images[sample_index].OutputImageFromKernel[0];

        a = __softmax(m_z);
    }
    else if(__funcType == ActuationFuncType::Tanh)
    {
        a = __tanh(v);
        //cout<<"tanh"<<endl;
    }
    else if(__funcType == ActuationFuncType::Relu)
    {
        a = __relu(v);
        //cout<<"relu"<<endl;
    }
    else
        std::cout<<"Error: unsupported actuation function type."<<std::endl;

    if(a < -1) 
    {
        std::cout<<"Error: Neuron::UpdateA(int sample_index), a<-1? something wrong."
            <<endl;
        exit(0);
    }
    //cout<<"debug: sample_id: "<<sample_index<<": a = "<<a<<endl;
    //getchar();
    //__a.push_back(a);
    __a[sample_index] = a;
}

void Neuron::UpdateSigmaPrime(int sample_index)
{
    // update sigma^prime
    //if(__sigmaPrime.size() != __z.size()-1) 
    //{
    //std::cout<<"Error: computing sigma^prime needs z computed first."<<std::endl;
    //exit(0);
    //}
    //if(__sigmaPrime.size() != __a.size()-1) 
    //{
    //std::cout<<"Error: computing sigma^prime needs a computed first."<<std::endl;
    //exit(0);
    //}

    //double a = __a.back();
    //double z = __z.back();
    double a = __a[sample_index];
    double z = __z[sample_index];
    double sigma_prime = -100; // theoretically, it must between [0, 1]

    if(__funcType == ActuationFuncType::Sigmoid || __funcType == ActuationFuncType::SoftMax) 
    {
        // the \partial a over \partial z is the same for sigmoid and softmax
        // one can easily prove it
        double diff = 1. - a;
        if( diff < 1e-10) diff = 0.; // to get rid of float number precision issues
        sigma_prime = diff*a;
    }
    else if(__funcType == ActuationFuncType::Tanh) 
    {
        sigma_prime = ( 2. / (exp(z) + exp(-z)) ) * ( 2. / (exp(z) + exp(-z)) );
    }
    else if(__funcType == ActuationFuncType::Relu) 
    {
        if( z > 0) sigma_prime = 1.;
        else sigma_prime = 0;
    }
    else
        std::cout<<"Error: unsupported actuation function type in direvative."<<std::endl;

    if(sigma_prime < 0.)
    {
        std::cout<<"Error: Neuron::UpdateSigmaPrime: sigma_prime<0? something wrong."
            <<endl;
        std::cout<<"    Actuation function type: "<<__funcType<<std::endl;
        std::cout<<"    Z: "<<z<<std::endl;
        std::cout<<"    a: "<<a<<std::endl;
        std::cout<<"    sigma prime: "<<sigma_prime<<std::endl;
        exit(0);
    }

    //__sigmaPrime.push_back(sigma_prime);
    __sigmaPrime[sample_index] = sigma_prime;
}

void Neuron::UpdateDelta(int sample_index)
{
    // update delta for current layer
    if(__layer->GetType() == LayerType::output)
    {
        UpdateDeltaOutputLayer(sample_index);
    } 
    else if(__layer->GetType() == LayerType::fullyConnected)
    {
        UpdateDeltaFC(sample_index);
    } 
    else if(__layer->GetType() == LayerType::cnn)
    {
        UpdateDeltaCNN(sample_index);
    } 
    else if(__layer->GetType() == LayerType::pooling)
    {
        UpdateDeltaPooling(sample_index);
    } 
    else 
    {
        std::cout<<"Error: unsupported layer type."<<std::endl;
        exit(0);
    }
}

void Neuron::UpdateDeltaOutputLayer(int sample_index)
{
    // back propagation delta for output layer
    if(__sigmaPrime.size() <= 0) 
    {
        std::cout<<"Error: Neuron::UpdateDeltaOutputLayer() computing delta needs sigma^prime computed first."<<std::endl;
        std::cout<<"        "<<__delta.size()<<" deltas, "<<__sigmaPrime.size()<<" sigma^primes"<<endl;
        exit(0);
    }
    assert(__a.size() == __sigmaPrime.size()); // make sure all have been updated
    auto & labels = __layer->GetDataInterface()->GetCurrentBatchLabel();
    assert(labels.size() == __a.size()); // make sure all samples have been processed

    //size_t batch_size = __a.size();

    // check
    auto dim = labels[0].Dimension(); // dim.first is neuron row number, dim.second is neuron collum number
    assert(__coord.i < dim.first);
    assert(dim.second == 1);
    //cout<<"neuron coord: "<<__coord<<endl;
    //assert(__coord.j == 0);

    //cout<<"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"<<endl;
    //cout<<"sample index: "<<sample_index<<endl;
    //cout<<"neuron id: "<<__neuron_id<<endl;
    //Print();

    auto cost_func_type = __layer->GetCostFuncType(); // for $\partial C over \partial a$

    // label for current sample
    Matrix & label_for_current_sample = labels[sample_index]; 
    //cout<<label_for_current_sample<<endl;
    // expected value for current sample current neuron
    float y_i = label_for_current_sample[__coord.i][0];
    //cout<<"---"<<y_i<<endl;

    // a value for current sample
    float a_i = __a[sample_index];
    //cout<<a_i<<endl;

    // sigma^\prime for current sample
    //float sigma_prime_i = __sigmaPrime[sample_index]; // obsolete
    //cout<<"---"<<sigma_prime_i<<endl;

    // solve for dC/da, which is dependent on the type of cost function
    float delta = 0;
    if(cost_func_type == CostFuncType::cross_entropy)
    {
        delta = a_i  - y_i;
    }
    else 
    {
        // other types of cost function: to be implemented
        delta = a_i - y_i; 
    }

    // save delta for this batch this neuron
    __delta[sample_index] = delta;
    //cout<<"here..."<<endl;
}


void Neuron::UpdateDeltaFC(int sample_index)
{
    // back propagation delta for fully connected layer
    if(__sigmaPrime.size() <= 0) 
    {
        std::cout<<"Error: Neuron::UpdateDeltaFC() computing delta needs sigma^prime computed first."<<std::endl;
        std::cout<<"        "<<__delta.size()<<" deltas, "<<__sigmaPrime.size()<<" sigma^primes"<<std::endl;
        exit(0);
    }
    //cout<<"here..... Neuron::UpdateDeltaFC()"<<endl;

    Layer* __nextLayer = __layer->GetNextLayer();

    auto & __deltaNext = __nextLayer->GetImagesActiveDelta(); // no vectorization needed, b/c "fc->cnn" type connection is not used in backward direction
    //cout<<"delta images batch size: "<<__deltaNext.size()<<endl;
    Images & image_delta_Next = __deltaNext[sample_index]; // get current sample delta
    std::vector<Matrix> &deltaNext = image_delta_Next.OutputImageFromKernel;
    if( deltaNext.size() != 1 ) 
    {
        std::cout<<"Error: Delta matrix dimension not match in FC layer"<<std::endl;
        exit(0);
    }
    Matrix & delta = deltaNext[0];
    //cout<<delta<<endl;

    //cout<<"sample id: "<<sample_index<<endl;

    auto wv = __nextLayer->GetWeightMatrix();
    /* 
    // 1) method 1
    Matrix w = Matrix::ConcatenateMatrixByI(wv); // for large matrix, this line is too time consuming, use method 2 instead

    auto w_dim = w.Dimension();
    if( w_dim.second < __coord.i ) 
    {
        std::cout<<"Error: weight matrix dimension not match in FC layer"<<std::endl;
        std::cout<<"Number of kernels: "<<wv->size()<<std::endl;
        std::cout<<"Neuron coord: "<<__coord<<std::endl;
        exit(0);
    }

    // back propogate delta
    w = w.Transpose();

    //cout<<w<<endl;
    auto dim = w.Dimension();
    w = w.GetSection(__coord.i, __coord.i+1, 0, dim.second);

    Matrix deltaCurrentLayer = w*delta;
    if(deltaCurrentLayer.Dimension().first != 1 || deltaCurrentLayer.Dimension().second != 1) 
    {
        std::cout<<"Error: back propagation delta, matrix dimension not match in FC layer."<<std::endl;
        exit(0);
    }
    */

    // 2) method 2
    auto delta_dim = delta.Dimension();
    assert(delta_dim.second == 1);
    double delta_current = 0;
    for(size_t i=0;i<delta_dim.first;i++)
    {
        delta_current += delta[i][0] * ((*wv)[i])[0][__coord.i];
    }

    // get sigma^\prime for current sample
    double s_prime = __sigmaPrime[sample_index];

    //double v = deltaCurrentLayer[0][0]; // method 1
    double v = delta_current; // method 2
    v = v*s_prime;

    __delta[sample_index] = v;

    //cout<<"here..... Neuron::UpdateDeltaFC()   end"<<endl;
}

static std::pair<size_t, size_t> mappedCoordsInVector(const std::pair<size_t, size_t> &kernel_dim, const NeuronCoord & coord)
{
    // background: after vectorization, tensors will be formed into a big vector
    //   each element in tensors will have a corresponding coord in the vector
    //
    // this function return the corresponding coordinates in that vector

    size_t total_elements_in_kernel = kernel_dim.first * kernel_dim.second;
    // nth_kernel should start from 0
    size_t i = coord.k * total_elements_in_kernel + coord.i * kernel_dim.first + coord.j;

    // vector has only one collum, so j = 0
    return std::pair<size_t, size_t>(i, 0);
}

void Neuron::UpdateDeltaCNN(int sample_index)
{
    // back propagate delta for cnn layer
    float delta_for_current_neuron_in_current_sample = 0.; // save results.

    // sigma prime of current neuron in current sample
    if(__sigmaPrime.size()<=0)
    {
        std::cout<<"Error: computing delta needs sigma^prime computed first."<<std::endl;
        exit(0);
    }
    double _sigma_prime = __sigmaPrime[sample_index];

    // next layer delta images
    Layer* __nextLayer = __layer->GetNextLayer();
    std::vector<Images> & deltaVecNext = __nextLayer->GetImagesActiveDelta(); // for 2D layer, images are all active, drop out happens on kernel

    // next layer weight matrix
    //auto weightVecNext = __nextLayer->GetWeightMatrix(); // obsolete, moved into subsection

    // output image size for current layer
    auto output_image_size_for_current_layer = __layer->GetOutputImageSize();

    if(__nextLayer->GetType() != LayerType::cnn && __nextLayer->GetType() != LayerType::pooling)
    {
        // next layer is fc or output, 1 dimensional
        assert(deltaVecNext[sample_index].OutputImageFromKernel.size() == 1);
        // delta image from next layer
        Matrix & m_delta_next_layer = deltaVecNext[sample_index].OutputImageFromKernel[0];
        // weight matrix from next layer
        std::vector<Matrix> *v_w_next_layer = __nextLayer->GetWeightMatrix();
        // make sure dimension match
        size_t n_active_neurons_in_next_layer = m_delta_next_layer.Dimension().first;
        assert(m_delta_next_layer.Dimension().second == 1);
        assert(v_w_next_layer->size() == n_active_neurons_in_next_layer);

        /* 
        // method 1 (this method should be used in layer level, otherwise too time consuming)
        Matrix w_next_layer = Matrix::ConcatenateMatrixByI(*v_w_next_layer);
        Matrix w_next_layer_T = w_next_layer.Transpose();

        Matrix m_delta_current_layer = w_next_layer_T * m_delta_next_layer;

        Images delta_image_current_layer_cache;
        delta_image_current_layer_cache.OutputImageFromKernel.push_back(m_delta_current_layer);

        Images delta_image_current_layer = 
        delta_image_current_layer_cache.Tensorization(output_image_size_for_current_layer.first, output_image_size_for_current_layer.second);

        delta_for_current_neuron_in_current_sample = (delta_image_current_layer.OutputImageFromKernel[__coord.k])[__coord.i][__coord.j] * _sigma_prime;
        cout<<"sample id: "<<sample_index<<" delta: "<<delta_for_current_neuron_in_current_sample<<endl;
        */	

        // method 2 (compute only for current neuron, not the whole layer in method 1)
        auto mapped_coords_in_vec = mappedCoordsInVector(output_image_size_for_current_layer, __coord);
        double tmp = 0;
        for(size_t iii = 0; iii<n_active_neurons_in_next_layer; iii++)
        {
            tmp += ((*v_w_next_layer)[iii])[0][mapped_coords_in_vec.first] * m_delta_next_layer[iii][0];
        }
        delta_for_current_neuron_in_current_sample = tmp * _sigma_prime;

        // already checked: method 1 and method 2 have the same results
        //cout<<"sample id: "<<sample_index<<" delta: "<<delta_for_current_neuron_in_current_sample<<endl;
    }
    else if(__nextLayer->GetType() == LayerType::cnn )
    {
        // next layer is cnn, 2 dimensional
        // vector of delta matrix from next layer
        std::vector<Matrix> & v_m_delta_next_layer = deltaVecNext[sample_index].OutputImageFromKernel;
        // vector of weight matrix from next layer
        std::vector<Matrix> * v_w_next_layer = __nextLayer->GetWeightMatrix();

        size_t C_next = v_m_delta_next_layer.size();
        assert(C_next == v_w_next_layer->size()); // make sure number of kernels = number of delta matrix in next layer

        // back propagation
        double tmp = 0;
        for(size_t d=0;d<C_next;d++) // sum all kernels
        { 
            // get d^th delta and weight matrix of next layer
            Matrix & delta = v_m_delta_next_layer[d];
            Matrix & weight = (*v_w_next_layer)[d];

            auto weight_dimension = weight.Dimension();
            auto delta_dimension =  delta.Dimension();

            for(size_t p=0; p<weight_dimension.first;p++){
                for(size_t q=0;q<weight_dimension.second;q++)
                {
                    double tmp_d = 0;
                    int delta_index_p = __coord.i - p;
                    int delta_index_q = __coord.j - q;
                    if(delta_index_p >= 0 && delta_index_q >= 0 && delta_index_p<(int)delta_dimension.first && delta_index_q<(int)delta_dimension.second )
                        tmp_d = delta[delta_index_p][delta_index_q];
                    double tmp_w = weight[p][q];
                    tmp += tmp_d * tmp_w;
                }
            }
        }

        // multiply  sigma_prime
        delta_for_current_neuron_in_current_sample = tmp * _sigma_prime;
    }
    else if(__nextLayer->GetType() == LayerType::pooling)
    {
        // next layer is pooling 
        // number of  delta images from next layer should = number of kernels in this layer
        assert(deltaVecNext[sample_index].OutputImageFromKernel.size() == __layer->GetWeightMatrix()->size());

        // --) directly get delta image matrix in next layer for current kernel (for pooling, kernels have 1-to-1 mapping relationship)
        Matrix & m_delta_next_layer = deltaVecNext[sample_index].OutputImageFromKernel[__coord.k];

        // --) get delta value corresponds to current neuron
        //     step 1) from neuron element coord get the corresponding coord in pooling image
        auto k_dim = __nextLayer->GetKernelDimensionCNN(); // get pooling kernel dimension from next layer
        int x_coord_in_pooling = (int)__coord.i / (int) k_dim.first;
        int y_coord_in_pooling = (int)__coord.j / (int) k_dim.second;
        assert(x_coord_in_pooling < (int)m_delta_next_layer.Dimension().first);
        assert(y_coord_in_pooling < (int)m_delta_next_layer.Dimension().second);

        float delta_candidate = m_delta_next_layer[x_coord_in_pooling][y_coord_in_pooling];

        // --) then check 'A' value of current neuron,
        if(__nextLayer->GetPoolingMethod() == PoolingMethod::Max)
        {
            // check if current neuron is the max one
            auto & v_images_in_current_layer = __layer->GetImagesActiveA();
            Images & images_in_current_layer =  v_images_in_current_layer[sample_index];
            Matrix & image_current_kernel = images_in_current_layer.OutputImageFromKernel[__coord.k];

            // get the max element in the covered section
            float max_a_sec = image_current_kernel.MaxInSectionWithPadding(x_coord_in_pooling*k_dim.first, (x_coord_in_pooling+1)*k_dim.first,
                    y_coord_in_pooling*k_dim.second, (y_coord_in_pooling+1)*k_dim.second);

            float a_current_neuron = __a[sample_index];

            if(a_current_neuron >= max_a_sec) 
                delta_for_current_neuron_in_current_sample = delta_candidate; // current neuron is the max one (the one being used)
            else 
                delta_for_current_neuron_in_current_sample = 0; // current neuron is not the max one (not used)
        }
        else if(__nextLayer->GetPoolingMethod() == PoolingMethod::Average)
        {
            delta_for_current_neuron_in_current_sample = delta_candidate;
        }
        else 
        {
            std::cout<<__func__<<" Error: undefined pooling method, something is wrong."
                <<endl;
            exit(0);
        }

        //cout<<"pooling->cnn backpropagation: to be tested..."<<endl;
    }
    else
    {
        std::cout<<__func__<<" Error: CNN layer followed by unknow type of layer, something is wrong."
            <<std::endl;
        exit(0);
    }

    __delta[sample_index] = delta_for_current_neuron_in_current_sample;
}

void Neuron::UpdateDeltaPooling(int sample_index)
{
    // back propagate delta for pooling layer
    // this layer works the same with cnn layer; should be merged with UpdateDeltaCNN
    if(__sigmaPrime.size() <= 0) 
    {
        // error
        std::cout<<__func__<<" Error: something wrong happend in error back propagation for pooling layer."
            <<std::endl;
        std::cout<<"       sigmaPrime matrix should be already calculated."<<std::endl;
        exit(0);
    }
    double _sigma_prime = __sigmaPrime[sample_index];
    double delta_for_this_neuron = 0.;

    // get delta images from its next layer
    Layer* __nextLayer = __layer->GetNextLayer();
    std::vector<Images> & deltaVecNext = __nextLayer->GetImagesActiveDelta(); // for 2D layer, images are all active, drop out happens on kernel

    auto outputImageSizeForCurrentLayer = __layer->GetOutputImageSize(); // output image dimension of current layer

    if(__nextLayer->GetType() != LayerType::cnn && __nextLayer->GetType()!= LayerType::pooling)
    {
        // next layer is a 1D layer
        assert(deltaVecNext[sample_index].OutputImageFromKernel.size() == 1);
        // delta image from next layer
        Matrix & m_delta_next_layer = deltaVecNext[sample_index].OutputImageFromKernel[0];
        // weight matrix from next layer
        std::vector<Matrix>* v_w_next_layer = __nextLayer->GetWeightMatrix();
        // make sure dimension match
        size_t n_active_neurons_in_next_layer = m_delta_next_layer.Dimension().first;
        assert(m_delta_next_layer.Dimension().second == 1);
        assert(v_w_next_layer->size() == n_active_neurons_in_next_layer);

        // backpropagation delta; use method 2
        auto mapped_coords_in_vec = mappedCoordsInVector(outputImageSizeForCurrentLayer, __coord);
        double tmp = 0;
        for(size_t iii= 0;iii<n_active_neurons_in_next_layer;iii++)
        {
            tmp += ((*v_w_next_layer)[iii])[0][mapped_coords_in_vec.first] * m_delta_next_layer[iii][0];
        }
        delta_for_this_neuron = tmp * _sigma_prime;
    }
    else if(__nextLayer->GetType() == LayerType::cnn)
    {
        // next layer is cnn, 2 dimensional
        // vector of delta matrix from next layer
        std::vector<Matrix> & v_m_delta_next_layer = deltaVecNext[sample_index].OutputImageFromKernel;
        // vector of weight matrix from next layer
        std::vector<Matrix> * v_w_next_layer = __nextLayer->GetWeightMatrix();

        size_t C_next = v_m_delta_next_layer.size();
        assert(C_next == v_w_next_layer->size()); // make sure number of kernels = number of delta matrix in next layer

        // back propagation
        double tmp = 0;
        for(size_t d=0;d<C_next;d++) // sum all kernels
        { 
            // get d^th delta and weight matrix of next layer
            Matrix & delta = v_m_delta_next_layer[d];
            Matrix & weight = (*v_w_next_layer)[d];

            auto weight_dimension = weight.Dimension();
            auto delta_dimension =  delta.Dimension();

            for(size_t p=0; p<weight_dimension.first;p++){
                for(size_t q=0;q<weight_dimension.second;q++)
                {
                    double tmp_d = 0;
                    int delta_index_p = __coord.i - p;
                    int delta_index_q = __coord.j - q;
                    if(delta_index_p >= 0 && delta_index_q >= 0 && delta_index_p<(int)delta_dimension.first && delta_index_q<(int)delta_dimension.second )
                        tmp_d = delta[delta_index_p][delta_index_q];
                    double tmp_w = weight[p][q];
                    tmp += tmp_d * tmp_w;
                }
            }
        }

        // multiply  sigma_prime
        delta_for_this_neuron = tmp * _sigma_prime;
    }
    else if(__nextLayer->GetType() == LayerType::pooling)
    {
        // next layer is also pooling : pooling->pooling (not practical, implement for code completeness)
        // number of  delta images from next layer should = number of kernels in this layer
        std::cout<<__func__<<"Error: pooling->pooling connection is meaningless, you can increase kernel dimension instead."
            <<std::endl;
        exit(0);
    }
    else
    {
        std::cout<<__func__<<" Error: pooling layer followed by unknow type of layer, something is wrong."
            <<std::endl;
        exit(0);
    }

    // for average pooling, give the same error to each neuron where it's related.
    __delta[sample_index] = delta_for_this_neuron;
}

void Neuron::ReplaceZ(int sample_id, double z_normalized)
{
    // a helper function for batch normalization process
    // replace the un-normalized z with normalized z

    __z[sample_id] = z_normalized;
}

void Neuron::SetCoord(size_t i, size_t j, size_t k)
{
    __coord.i = i; __coord.j = j; __coord.k = k;
}

void Neuron::SetCoordI(size_t v)
{
    __coord.i = v;
}

void Neuron::SetCoordJ(size_t v)
{
    __coord.j = v;
}

void Neuron::SetCoordK(size_t v)
{
    __coord.k = v;
}

void Neuron::SetCoord(NeuronCoord c)
{
    __coord = c;
}

std::vector<double>& Neuron::GetAVector()
{
    return __a;
}

std::vector<double>& Neuron::GetDeltaVector()
{
    return __delta;
}

std::vector<double>& Neuron::GetZVector()
{
    return __z;
}


std::vector<double>& Neuron::GetSigmaPrimeVector()
{
    return __sigmaPrime;
}



NeuronCoord Neuron::GetCoord()
{
    // get neuron coord
    return __coord;
}

void Neuron::ClearPreviousBatch()
{
    __a.clear();           // necessary, since resize won't overwrite old contents
    __delta.clear();       // 
    __z.clear();           // 
    __sigmaPrime.clear();  // 

    if(__layer == nullptr)
    {
        std::cout<<"Error: Neuron::ClearPreviousBatch(): the layer info has not been set for this neuron."
            <<endl;
        exit(0);
    }
    int batch_size = __layer->GetBatchSize();
    __a.resize(batch_size); // reset a
    __delta.resize(batch_size); // reset delta
    __z.resize(batch_size); // reset z
    __sigmaPrime.resize(batch_size);
}

void Neuron::Print()
{
    std::cout<<"--------------------------------------"<<std::endl;
    std::cout<<"neuron id: "<<__neuron_id<<std::endl;
    std::cout<<"active status: "<<__active_status<<std::endl;
    if(!__active_status) return;
    std::cout<<"w matrix: "<<std::endl;
    std::cout<<(*__w);
    std::cout<<"bias: "<<std::endl;
    std::cout<<(*__b);
    std::cout<<"Neruon Coord: "<<endl;
    std::cout<<__coord<<endl;
}
