#include "Network.h"

#include "DataInterface.h"
#include "Layer.h"
#include "ConstructLayer.h"

#include <iostream>
#include <chrono>

using namespace std;

Network::Network()
{
    // place holder
}

Network::~Network()
{
    // place holder
}

void Network::Init()
{
    // construct all layers
    //ConstructLayers();

    // set number of epochs
    //__numberOfEpoch = 10;
}

/*
void Network::ConstructLayers(TrainingType training_type)
{
    // This is an working example showing how to construct Networks, and the nominal values for hyper parameters 
    // such as learning_rate, regularization factor etc.
    // Network structure: {Image->Input->CNN->pooling->CNN->pooling->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    //DataInterface *data_interface = new DataInterface("simulation_data/data_signal_train.dat", "simulation_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(10, 10), 200);
    DataInterface *data_interface = new DataInterface("simulation_data/data_signal_train.dat", "simulation_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(10, 10), 10);
    //DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(100, 100), 500);

    //TrainingType resume_or_new_training = training_type;
    //TrainingType resume_or_new_training = TrainingType::ResumeTraining;
    TrainingType resume_or_new_training = TrainingType::NewTraining;

    double learning_rate = 0.06; // worked: 0.6
    double regularization_factor = 0.01; // worked: 0.1
    // for Adam optimizer, the best values seems to be: learning_rate = 0.06; regularization_factor = 0.01;
    //           even though the paper (https://arxiv.org/pdf/1412.6980.pdf) suggested learning_rate to be 0.01;
    //           the smaller the regularization factor, the better the training result seems to be,
    //           which makes sense, smaller regularization factor means the training is more sensitive to 
    //           data peculiarities, thus training should seem more better; 
    //           bigger regularization factor means the training is more foucus on 
    //           getting small consistent weights, the prediction should be better when training converged.

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, resume_or_new_training, false);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    //       input layer does not need next layer connection, b/c it does not participate in backpropagation
    layer_input->Init();

    // 4) middle layer 3 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 1, 
	    std::pair<size_t, size_t>(4, 4), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Sigmoid, resume_or_new_training, false);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

     // 4) middle layer 3 : cnn layer ID=3
    LayerParameterList p_list3(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 1, 
	    std::pair<size_t, size_t>(4, 4), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Sigmoid, resume_or_new_training, false);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(l1);
    l3->Init();

     // 5) output layer ID = 6
    LayerParameterList p_list6(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0., Regularization::L2, regularization_factor, ActuationFuncType::SoftMax, resume_or_new_training, false);
    //LayerParameterList p_list6(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
//	    std::pair<size_t, size_t>(0, 0), 0., false, 0, 0., Regularization::L2, 0., ActuationFuncType::SoftMax, resume_or_new_training, false);
    Layer* layer_output = new ConstructLayer(p_list6);
    layer_output -> SetPrevLayer(l3);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l3);
    l3->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleAndOutputLayers.push_back(l1); // must be pushed in order
    __middleAndOutputLayers.push_back(l3);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;


    // save all initialized weights and bias
    //__inputLayer->SaveTrainedWeightsAndBias();
    //for(auto &i: __middleAndOutputLayers)
    //    i->SaveTrainedWeightsAndBias();
}
*/

void Network::ConstructLayers(TrainingType training_type)
{
    // Network structure: {Image->Input->CNN->pooling->CNN->pooling->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("simulation_data/data_signal_train.dat", "simulation_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(50, 50), 200);
    //DataInterface *data_interface = new DataInterface("test_data/data_signal_train.dat", "test_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(100, 100), 500);

    //TrainingType resume_or_new_training = training_type;
    //TrainingType resume_or_new_training = TrainingType::ResumeTraining;
    TrainingType resume_or_new_training = TrainingType::NewTraining;
    //double learning_rate = 0.01; //0.06;
    double learning_rate = 0.06; //0.06; // if use batch normalization, one can use relatively large learning rate
    double regularization_factor = 0.005;

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, resume_or_new_training, false);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    //       input layer does not need next layer connection, b/c it does not participate in backpropagation
    layer_input->Init();

    // 4) middle layer 3 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(3, 3), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training, true);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();
  
    // 4) middle layer 2 : pooling layer ID=2
    LayerParameterList p_list2(LayerType::pooling, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(3, 3), learning_rate, false, 0, 0., Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training, false);
    Layer *l2 = new ConstructLayer(p_list2);
    l2->SetPrevLayer(l1);
    l2->Init();
 
    // 4) middle layer 3 : cnn layer ID=3
    LayerParameterList p_list3(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(3, 3), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training, true);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(l2);
    l3->Init();
  
    // 4) middle layer 2 : pooling layer ID=4
    LayerParameterList p_list4(LayerType::pooling, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(3, 3), learning_rate, false, 0, 0., Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training, false);
    Layer *l4 = new ConstructLayer(p_list4);
    l4->SetPrevLayer(l3);
    l4->Init();

    // 4) middle layer 1 : fc layer ID=5
    LayerParameterList p_list5(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 7, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training, true);
    Layer *l5 = new ConstructLayer(p_list5);
    l5->SetPrevLayer(l4);
    l5->Init();

    // 5) output layer ID = 6
    LayerParameterList p_list6(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0., Regularization::L2, regularization_factor, ActuationFuncType::SoftMax, resume_or_new_training, true);
    Layer* layer_output = new ConstructLayer(p_list6);
    layer_output -> SetPrevLayer(l5);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l2);
    l2->SetNextLayer(l3);
    l3->SetNextLayer(l4);
    l4->SetNextLayer(l5);
    l5->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleAndOutputLayers.push_back(l1); // must be pushed in order
    __middleAndOutputLayers.push_back(l2); // must be pushed in order
    __middleAndOutputLayers.push_back(l3);
    __middleAndOutputLayers.push_back(l4);
    __middleAndOutputLayers.push_back(l5);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;
}


/*
void Network::ConstructLayers(TrainingType _training_type) // test purelly fully connected
{
    // Network structure: {Image->Input->FC->FC->FC->Output}
    TrainingType training_type = _training_type;
    training_type = TrainingType::NewTraining;
    double learning_rate = 0.1;

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("simulation_data/data_signal_train.dat", "simulation_data/data_cosmic_train.dat", LayerDimension::_1D, std::pair<int, int>(900, 1), 200);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_1D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, training_type, false);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list3(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 10, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 2, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, training_type, false);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(layer_input);
    l3->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax, training_type, false);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l3);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    //l1->SetNextLayer(l2);
    //l2->SetNextLayer(l3);
    l3->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    //__middleAndOutputLayers.push_back(l1); // must be pushed in order
    //__middleAndOutputLayers.push_back(l2); // must be pushed in order
    __middleAndOutputLayers.push_back(l3);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;
}
*/

/*
void Network::ConstructLayers(TrainingType trtp) // test fully connected + cnn
{
    // Network structure: {Image->Input->FC->FC->FC->Output}

    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("simulation_data/data_signal_train.dat", "simulation_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(10, 10), 200);

    //TrainingType training_type = trtp;
    TrainingType training_type = TrainingType::NewTraining;
    //TrainingType training_type = TrainingType::ResumeTraining;
    double learning_rate = 0.6;

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, training_type, false);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // 4) middle layer 1 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 3, 
	    std::pair<size_t, size_t>(3, 3), learning_rate, false, 3, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, training_type, false);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list6(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 14, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 3, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, training_type, false);
    Layer *l6 = new ConstructLayer(p_list6);
    l6->SetPrevLayer(l1);
    l6->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax, training_type, false);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l6);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l6);
    //l2->SetNextLayer(l3);
    l6->SetNextLayer(layer_output); // This line is ugly, to be improved

    // 7) save all constructed layers
    __inputLayer = layer_input;
    __outputLayer = layer_output;
    __middleAndOutputLayers.push_back(l1); // must be pushed in order
    //__middleAndOutputLayers.push_back(l2); // must be pushed in order
    __middleAndOutputLayers.push_back(l6);
    __middleAndOutputLayers.push_back(layer_output);
    //cout<<"total number of layers: "<<__middleAndOutputLayers.size()<<endl;
    __dataInterface = data_interface;
}
*/

void Network::Train()
{
    // construct layers
    TrainingType training_type = TrainingType::NewTraining;
    ConstructLayers(training_type);

    __numberOfEpoch = 50; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    for(int i=0;i<__numberOfEpoch;i++)
    {
        std::cout<<"[------]Number of epoch: "<<i<<"/"<<__numberOfEpoch<<endl;
        UpdateEpoch();
    }

    // check accuracy track of training
    cout<<"accuracy: "<<endl;
    std::vector<double> & accuracy = __outputLayer->GetAccuracyForBatches();
    for(auto &i: accuracy)
        cout<<i<<",   "<<endl;

    // check cost track of training
    cout<<"cost: "<<endl;
    std::vector<double> &cost = __outputLayer->GetCostForBatches();
    for(auto &i: cost)
        cout<<i<<", "<<endl;

    // after finished training, save all trained weights and bias
    __inputLayer->SaveTrainedWeightsAndBias();
    for(auto &i: __middleAndOutputLayers)
        i->SaveTrainedWeightsAndBias();
}

void Network::UpdateEpoch()
{
    int numberofBatches = __dataInterface -> GetNumberOfBatches();
    cout<<"......Info: "<<numberofBatches<<" batches in this epoch"<<endl;
    numberofBatches = 1; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQqq

    __dataInterface->Reset();
    __dataInterface->Shuffle();

    // initializations for epoch
    for(auto &i: __middleAndOutputLayers)
    {
	i->EpochInit();
    }

    for(int i=0;i<numberofBatches;i++)
    {
        cout<<"......... training for batch: "<<i<<"/"<<numberofBatches<<endl;
        UpdateBatch();
    }
}

void Network::UpdateBatch()
{
    // initializations for batch
    for(auto &i: __middleAndOutputLayers)
        i->BatchInit();

    auto t1 = std::chrono::high_resolution_clock::now();
    ForwardPropagateForBatch();
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto dt1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
    //std::cout<<"forward propagation cost: "<<dt1.count()<<" milliseconds"<<endl;

    BackwardPropagateForBatch();
    //auto t3 = std::chrono::high_resolution_clock::now();
    //auto dt2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2);
    //std::cout<<"backward propagation cost: "<<dt2.count()<<" milliseconds"<<endl;

    UpdateWeightsAndBiasForBatch();
    auto t4 = std::chrono::high_resolution_clock::now();
    auto dt3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t1);
    std::cout<<"        batch training cost: "<<dt3.count()<<" milliseconds"<<endl;

    // save training accuracy for batch
    __outputLayer -> SaveAccuracyAndCostForBatch();
}

void Network::ForwardPropagateForBatch()
{
    // prepare new batch data and label in Datainterface class
    __dataInterface->GetNewBatchData();
    __dataInterface->GetNewBatchLabel();

    // fill data to input layer
    __inputLayer->FillBatchDataToInputLayerA();

    // forward propagation
    for(auto &i: __middleAndOutputLayers)
    {
	i->ForwardPropagateForBatch();
    }
}

void Network::BackwardPropagateForBatch() 
{
    // backward
    int NLayers = __middleAndOutputLayers.size();

    for(int nlayer=NLayers-1; nlayer>=0; nlayer--)
	__middleAndOutputLayers[nlayer]->BackwardPropagateForBatch();

    /// no need for input layer
}

void Network::UpdateWeightsAndBiasForBatch()
{
    // update w&b for output and  middle layers
    for(auto &i: __middleAndOutputLayers)
	i->UpdateWeightsAndBias();
}

std::vector<Matrix> Network::Classify()
{
    std::vector<Matrix> res;
    ConstructLayers(TrainingType::ResumeTraining);

    __numberOfEpoch = 1; // test QQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    for(int i=0;i<__numberOfEpoch;i++)
    {
	std::cout<<"[------]Number of epoch: "<<i<<"/"<<__numberOfEpoch<<endl;
	UpdateEpoch();
    }

    // check accuracy 
    cout<<"accuracy: "<<endl;
    std::vector<double> & accuracy = __outputLayer->GetAccuracyForBatches();
    for(auto &i: accuracy)
	cout<<i<<",   "<<endl;

    // check cost 
    cout<<"cost: "<<endl;
    std::vector<double> &cost = __outputLayer->GetCostForBatches();
    for(auto &i: cost)
	cout<<i<<", "<<endl;

    return res;
}
