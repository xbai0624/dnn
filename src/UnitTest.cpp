#include "UnitTest.h"

#include <iostream>

#include "DataInterface.h"
#include "Layer.h" // test layer
#include "ConstructLayer.h"

using namespace std;

UnitTest::UnitTest()
{
    // reserved
}

UnitTest::~UnitTest()
{
    // reserved
}

void UnitTest::Test()
{
    //TestFilter2D();
    //TestImagesStruct();
    TestMatrix();

    //TestDNN();

    //TestCNN();
    //TestCNNWeightAndBiasEvolution();

    //TestCNNToPooling();
    //TestCNNToCNN();
}

void UnitTest::TestFilter2D()
{
/*
    Filter2D F(4, 4);
    cout<<F<<endl;

    Filter2D f = F;
    cout<<f<<endl;

    // test default assign
    vector<vector<bool>> b(4, vector<bool>(4, true));
    vector<vector<bool>> c;
    c = b;

    for(auto i: c)
    {
	for(auto j: i)
	    cout<<j<<", ";
	cout<<endl;
    }

    // test default assignment
    vector<Filter2D> VF;
    VF.resize(4, Filter2D(3, 3));
    vector<Filter2D> VFC;
    VFC = VF;

    for(auto &i: VFC)
        cout<<i<<endl;


    Filter2D notF = F.Opposite();
    cout<<"-------------------"<<endl;
    cout<<F<<endl;
    cout<<notF<<endl;
*/
    cout<<"++++++++++++++++++++++++++++++++++"<<endl;
    //Filter2D FF(100, 100, true);
    //vector<Filter2D> complete_set = FF.GenerateCompleteDropOutSet(3, 0.4);
    //for(auto &i: complete_set) cout<<i<<endl;

    Filter2D FF(10, 1, true);
    vector<Filter2D> complete_set = FF.GenerateCompleteDropOutSet(20, 0.5);
    for(auto &i: complete_set) 
	cout<<i<<endl;


    //Filter2D FF(500, 500, true);
    //vector<Filter2D> complete_set = FF.GenerateCompleteDropOutSet(4, 0.33333);
    //for(auto &i: complete_set)  cout<<i<<endl;
}


void UnitTest::TestImagesStruct()
{
    // test Images struct in Layer.h file

    // 1) empty test
    Images __images;
    //Images v_image0 = __images.Vectorization();

    // 2) vectorization functionality test
    cout<<"vectorization test"<<endl;
    for(int i=0;i<4;i++)
    {
	Matrix kernel(3,4);
	kernel.Random(); // fill matrix with random numbers
	__images.OutputImageFromKernel.push_back(kernel);
    }
    for(auto &i: __images.OutputImageFromKernel)
	cout<<i<<endl<<endl;

    // 2-1) test copy
    cout<<"test copy."<<endl;
    Images c_image = __images;
    for(auto &i: c_image.OutputImageFromKernel)
	cout<<i<<endl<<endl;
    Images v_image = __images.Vectorization();
    for(auto &i: v_image.OutputImageFromKernel)
	cout<<i<<endl<<endl;

    // 3) dimension not match test
    //Matrix k(2, 4, 0);
    //__images.OutputImageFromKernel.push_back(k);
    //v_image = __images.Vectorization();

    // 4) tensorization test
    cout<<"Tensorization test"<<endl;
    Images tensor_image = v_image.Tensorization(3, 4);
    for(auto &i: tensor_image.OutputImageFromKernel)
	cout<<i<<endl<<endl;
}


void UnitTest::TestMatrix()
{
/*
    Matrix m(4, 4);
    m.Random();

    cout<<"Test Matrix."<<endl;
    cout<<m<<endl;

    double v = m.MaxInSectionWithPadding(2, 6, 2, 6);
    cout<<v<<endl;

    cout<<"Test average in section with padding."<<endl;
    Matrix mm(4, 4, 1);
    cout<<mm<<endl;
    double vv = mm.AverageInSectionWithPadding(0, 6, 1, 6);
    cout<<vv<<endl;

    Matrix m1(3, 1, 0);
    Matrix m2(3, 1, 1);
    bool e = (m1 == m2);
    cout<<m1<<endl;
    cout<<m2<<endl;
    cout<<"matrix equal: "<<e<<endl;

    Matrix m(4, 10);
    for(int i=0;i<10;i++)
    {
	m.RandomGaus(0, 1./100);
	cout<<m<<endl;
    }

    Matrix m(4, 4, 0);
    m[3][2] = 1;
    cout<<m<<endl;
    Matrix mm = m.Normalization();
    cout<<mm<<endl;
*/
    vector<Matrix> vm;
    Matrix m1(3, 3, 1);
    Matrix m2(3, 3, 2);
    Matrix m3(3, 3, 1);
    Matrix m4(3, 3, 0);


    vm.push_back(m1);
    vm.push_back(m4);
    //vm.push_back(m2);
    //vm.push_back(m3);
    vm.push_back(m4);
    cout<<vm<<endl;

    Matrix::BatchNormalization(vm);
    cout<<vm<<endl;
}

void UnitTest::TestDNN()
{
    // setup a 1D data interface
    DataInterface *data_interface = new DataInterface("unit_test_data/data_signal.dat", "unit_test_data/data_cosmic.dat", LayerDimension::_1D);

    // setup a 1D input layer
    LayerParameterList p_list0(LayerType::input, LayerDimension::_1D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, 0.5, Regularization::Undefined, 0, ActuationFuncType::Sigmoid, TrainingType::NewTraining);
    Layer* layer_input = new ConstructLayer(p_list0);
    // NOTE: a data_interface class pointer must be passed to input layer before calling input_layer->Init() function
    //       because Initialization rely on data_interface
    layer_input->Init();

    // setup a FC layer
    LayerParameterList p_list3(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 5, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, true, 2, 0.5, Regularization::L2, 0.1, ActuationFuncType::Sigmoid, TrainingType::NewTraining);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(layer_input);
    l3->Init();

    // setup an output layer
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::SoftMax, TrainingType::NewTraining);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l3);
    layer_output -> Init();

    // connect all  layers
    l3->SetNextLayer(layer_output); // This line is ugly, to be improved


    // loop epoch
    for(int epoch = 0;epoch<1;epoch++) // only 1 epoch
    { //
	// loop batch
	for(int nbatch = 0;nbatch<1;nbatch++) // only 1 batch
	{ //

	    // test start here
	    l3->BatchInit();
	    layer_output->BatchInit();

	    data_interface -> GetNewBatchData();
	    data_interface -> GetNewBatchLabel();
	    layer_input -> FillBatchDataToInputLayerA();

	    size_t sample_size = layer_input->GetBatchSize();
	    cout<<"sample size: "<<sample_size<<endl;

	    auto show_layer_in_forward = [&](Layer *l, size_t id)
	    {
		// show full a image
		cout<<"===layer=layer=layer=layer=layer=layer=layer=layer=layer==="<<endl;
		cout<<"layer id: "<<l->GetID()<<", "<<l->GetType()<<endl;

		cout<<"original w Matrix: "<<endl;
		for(auto &i: *(l->GetWeightMatrixOriginal()))
		    cout<<i<<endl;

		//cout<<"active w Matrix: "<<endl;
		//for(auto &i: *(l->GetWeightMatrix()))
		//    cout<<i<<endl;

		cout<<"original bias vector: "<<endl;
		for(auto &i: *(l->GetBiasVectorOriginal()))
		    cout<<i<<endl;

		//cout<<"active bias vector: "<<endl;
		//for(auto &i: *(l->GetBiasVector()))
		//    cout<<i<<endl;

		//cout<<"active Z images: "<<endl;
		//if((l->GetImagesActiveZ()).size() > 0)
		//    for(auto &i: (l->GetImagesActiveZ())[id].OutputImageFromKernel)
		//		cout<<i<<endl;

		cout<<"full Z images: "<<endl;
		if((l->GetImagesFullZ()).size()>0)
		    for(auto &i: (l->GetImagesFullZ())[id].OutputImageFromKernel)
			cout<<i<<endl;

		//cout<<"active a images: "<<endl;
		//if((l->GetImagesActiveA()).size() > 0)
		//    for(auto &i: (l->GetImagesActiveA())[id].OutputImageFromKernel)
		//	cout<<i<<endl;

		cout<<"full a images: "<<endl;
		if((l->GetImagesFullA()).size() > 0)
		    for(auto &i: (l->GetImagesFullA())[id].OutputImageFromKernel)
			cout<<i<<endl;
	    };


	    // loop sample forward direction
	    for(size_t sample_id = 0;sample_id < sample_size;sample_id++)
	    {
		cout<<"forward propagation sample: "<<sample_id<<endl;
		l3->ForwardPropagate_Z_ForSample(sample_id);
		l3->ForwardPropagate_SigmaPrimeAndA_ForSample(sample_id);
		layer_output->ForwardPropagate_Z_ForSample(sample_id);
		layer_output->ForwardPropagate_SigmaPrimeAndA_ForSample(sample_id);
	    }

	    //==============================================================
	    // use your eyes here.... check forward propagation
	    //==============================================================
	    for(size_t id=0;id<sample_size;id++)
	    {
		// check for each sample
		cout<<"%%%%%%%%%%%%%%%%%%%%%%% checking sample : "<<id<<" start %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;
		show_layer_in_forward(layer_input, id);
		show_layer_in_forward(l3, id);
		show_layer_in_forward(layer_output, id);
		cout<<"%%%%%%%%%%%%%%%%%%%%%%% checking sample : "<<id<<" end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<endl;
		getchar();
	    }
	    // check everything is saved properly after batch finished
	    cout<<"................ Check Batch Results ................."<<endl;
	    auto check_forward_batch_results= [&](Layer* layer) 
	    {
		cout<<"############## layer: "<<layer->GetID()<<" ##############"<<endl;
		cout<<"--------  full image Z: -------  "<<endl;
		for(auto &i: layer->GetImagesFullZ())
		{
		    cout<<"sample:"<<endl;
		    for(auto &j: i.OutputImageFromKernel)
		    {
			cout<<"    kernel: "<<endl;
			cout<<j<<endl;
		    }
		}

		cout<<"--------  full image A: -------  "<<endl;
		for(auto &i: layer->GetImagesFullA())
		{
		    cout<<"sample:"<<endl;
		    for(auto &j: i.OutputImageFromKernel)
		    {
			cout<<"    kernel: "<<endl;
			cout<<j<<endl;
		    }
		}

		cout<<"--------  full image SigmaPrime: -------  "<<endl;
		for(auto &i: layer->GetImagesFullSigmaPrime())
		{
		    cout<<"sample:"<<endl;
		    for(auto &j: i.OutputImageFromKernel)
		    {
			cout<<"    kernel: "<<endl;
			cout<<j<<endl;
		    }
		}

	    };

	    check_forward_batch_results(layer_input);
	    check_forward_batch_results(l3);
	    check_forward_batch_results(layer_output);

	    // loop sample backward direction
	    for(size_t sample_id = 0;sample_id < sample_size;sample_id++)
	    {
		cout<<"backward propagation sample: "<<sample_id<<endl;
		layer_output->BackwardPropagateForSample(sample_id);
		l3->BackwardPropagateForSample(sample_id);
	    }

	    // show backward batch results
	    auto show_layer_in_backward = [&](Layer* layer)
	    {
		cout<<"===layer=layer=layer=layer=layer=layer=layer=layer=layer==="<<endl;
		cout<<"layer id: "<<layer->GetID()<<", "<<layer->GetType()<<endl;

		cout<<"--------  full image Delta: -------  "<<endl;
		for(auto &i: layer->GetImagesFullDelta())
		{
		    cout<<"sample:"<<endl;
		    for(auto &j: i.OutputImageFromKernel)
		    {
			cout<<"    kernel: "<<endl;
			cout<<j<<endl;
		    }
		}
	    };

	    auto print_wb_matrix =[&](Layer *l)
	    {
		cout<<"W&B matrix of layer id: "<<l->GetID()<<", "<<l->GetType()<<endl;
		cout<<"original w Matrix: "<<endl;
		for(auto &i: *(l->GetWeightMatrixOriginal()))
		    cout<<i<<endl;

		cout<<"original bias vector: "<<endl;
		for(auto &i: *(l->GetBiasVectorOriginal()))
		    cout<<i<<endl;
	    };

	    //==============================================================
	    //     use your eyes here.... check backward propagation
	    //==============================================================
	    cout<<"######################### BACKWARDCHECK ########################"<<endl;
	    show_layer_in_backward(layer_output);
	    print_wb_matrix(layer_output);
	    show_layer_in_backward(l3);


	    // check weights and bias update
	    cout<<"show l3 previous layer a image: "<<endl;
	    cout<<"show l3 layer delta image: "<<endl;
	    l3->UpdateWeightsAndBias();

	    layer_output->UpdateWeightsAndBias();


	} //
    } //
}



void UnitTest::TestCNN()
{
    // step1: setup all layers.............................................................................................................
    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("unit_test_data/data_signal.dat", "unit_test_data/data_cosmic.dat", LayerDimension::_2D);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, TrainingType::NewTraining);
    Layer* layer_input = new ConstructLayer(p_list0);
    layer_input->Init();

    // 4) middle layer 1 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 2, 
	    std::pair<size_t, size_t>(2, 2), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, TrainingType::NewTraining);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list6(LayerType::fullyConnected, LayerDimension::_1D, data_interface, 10, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, TrainingType::NewTraining);
    Layer *l6 = new ConstructLayer(p_list6);
    l6->SetPrevLayer(l1);
    l6->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax, TrainingType::NewTraining);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l6);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l6);
    //l2->SetNextLayer(l3);
    l6->SetNextLayer(layer_output); // This line is ugly, to be improved


    // step 2: forward propagate
    data_interface -> Reset();
    layer_input->EpochInit();
    l1->EpochInit();
    l6->EpochInit();
    layer_output->EpochInit();

    //layer_input->BatchInit();
    l1->BatchInit();
    l6->BatchInit();
    layer_output->BatchInit();


    data_interface->GetNewBatchData();
    data_interface->GetNewBatchLabel();
    layer_input -> FillBatchDataToInputLayerA();


    int sample_size = data_interface->GetBatchSize();
    for(int i=0;i<sample_size;i++)
    {
	l1->ForwardPropagate_Z_ForSample(i);
	l1->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
	l6->ForwardPropagate_Z_ForSample(i);
	l6->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
	layer_output->ForwardPropagate_Z_ForSample(i);
	layer_output->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
    }

    // helper functions
    auto print_a_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'A' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullA();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };
    auto print_z_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'Z' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullZ();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };


    auto print_W_B_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::W&B matrix from layer: "<<l->GetID()<<endl;
	auto w_images = l->GetWeightMatrixOriginal();
	auto b_images = l->GetBiasVectorOriginal();
	assert((*w_images).size() == (*b_images).size());

	for(size_t kernel = 0;kernel<(*w_images).size();kernel++)
	{
	    cout<<"----kernel id: "<<kernel<<endl;
	    cout<<"w"<<kernel<<":"<<endl;
	    cout<<(*w_images)[kernel]<<endl;
	    cout<<"b"<<kernel<<":"<<endl;
	    cout<<(*b_images)[kernel]<<endl;
	}
    };

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_W_B_images(layer_input);
    print_z_images(layer_input);
    print_a_images(layer_input);

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_W_B_images(l1);
    //print_z_images(l1);
    //print_a_images(l1);

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_W_B_images(l6);
    //print_z_images(l6);
    //print_a_images(l6);

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_W_B_images(layer_output);
    //print_z_images(layer_output);
    //print_a_images(layer_output);


    // backward propagation
    for(int i=0;i<sample_size;i++)
    {
	// must inverse order
	layer_output->BackwardPropagateForSample(i);
	l6->BackwardPropagateForSample(i);
	l1->BackwardPropagateForSample(i);
    }


    auto print_sigmaPrime_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'SigmaPrime' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullSigmaPrime();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };

    auto print_delta_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'Delta' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullDelta();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };

    auto print_w_b_gradients = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'w & b' gradients from layer: "<<l->GetID()<<endl;
	auto w_images = l->GetWeightGradients();
	auto b_images = l->GetBiasGradients();
	assert(w_images.size() == b_images.size());
	for(size_t i=0;i<w_images.size();i++)
	{
	    cout<<"---------------------- sample id: "<<i<<endl;
	    int kernel_id = 0;
	    for(auto &j: w_images[i].OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    kernel_id = 0;
	    for(auto &j: b_images[i].OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	}
    };

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_a_images(layer_output);
    print_sigmaPrime_images(layer_output); 
    print_delta_images(layer_output);
    print_W_B_images(layer_output);

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_a_images(l6);
    //print_sigmaPrime_images(l6); 
    //print_delta_images(l6);
    //print_W_B_images(l6);


    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_a_images(l1);
    //print_sigmaPrime_images(l1); 
    //print_delta_images(l1);
    //print_W_B_images(l1);

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_a_images(layer_input); 

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_w_b_gradients(l1); 

    // update weights and gradients
    layer_output->UpdateWeightsAndBias();
    l6->UpdateWeightsAndBias();
    //cout<<"?????????????????"<<endl;
    l1->UpdateWeightsAndBias();

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_w_b_gradients(l1); 
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_W_B_images(l1);


}


void UnitTest::TestCNNToPooling()
{
    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("unit_test_data/data_signal.dat", "unit_test_data/data_cosmic.dat", LayerDimension::_2D);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, TrainingType::NewTraining);
    Layer* layer_input = new ConstructLayer(p_list0);
    layer_input->Init();

    // 4) middle layer 1 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 2, 
	    std::pair<size_t, size_t>(2, 2), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, TrainingType::NewTraining);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list6(LayerType::pooling, LayerDimension::_2D, data_interface, 0, 2, 
	    std::pair<size_t, size_t>(2, 2), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, TrainingType::NewTraining);
    Layer *l6 = new ConstructLayer(p_list6);
    l6->SetPrevLayer(l1);
    l6->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax, TrainingType::NewTraining);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l6);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l6);
    //l2->SetNextLayer(l3);
    l6->SetNextLayer(layer_output); // This line is ugly, to be improved


    // step 2: forward propagate
    data_interface -> Reset();
    layer_input->EpochInit();
    l1->EpochInit();
    l6->EpochInit();
    layer_output->EpochInit();

    //layer_input->BatchInit();
    l1->BatchInit();
    l6->BatchInit();
    layer_output->BatchInit();


    data_interface->GetNewBatchData();
    data_interface->GetNewBatchLabel();
    layer_input -> FillBatchDataToInputLayerA();


    // helper functions
    auto print_a_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'A' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullA();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };
    auto print_z_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'Z' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullZ();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };

    // forward propagation
    int sample_size = data_interface->GetBatchSize();
    for(int i=0;i<sample_size;i++)
    {
	l1->ForwardPropagate_Z_ForSample(i);
	l1->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
	
	l6->ForwardPropagate_Z_ForSample(i);
	l6->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
	
	layer_output->ForwardPropagate_Z_ForSample(i);
	layer_output->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
    }

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_a_images(l1);
    print_z_images(l1);
    //print_a_images(l6);
    //print_z_images(l6);


    // backward propagation
    for(int i=0;i<sample_size;i++)
    {
	// must inverse order
	layer_output->BackwardPropagateForSample(i);
	l6->BackwardPropagateForSample(i);
	l1->BackwardPropagateForSample(i);
    }

    auto print_delta_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'Delta' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullDelta();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_a_images(l1);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_a_images(l6);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_delta_images(l6);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_delta_images(l1);

}

void UnitTest::TestCNNToCNN()
{
    // 1) Data interface, this is a tool class, for data prepare
    DataInterface *data_interface = new DataInterface("unit_test_data/data_signal.dat", "unit_test_data/data_cosmic.dat", LayerDimension::_2D);

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), 0, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, TrainingType::NewTraining);
    Layer* layer_input = new ConstructLayer(p_list0);
    layer_input->Init();

    // 4) middle layer 1 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 2, 
	    std::pair<size_t, size_t>(2, 2), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, TrainingType::NewTraining);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

    // 4) middle layer 1 : fc layer ID=6
    LayerParameterList p_list6(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 2, 
	    std::pair<size_t, size_t>(2, 2), 0.1, false, 0, 0.5, Regularization::L2, 0.1, ActuationFuncType::Relu, TrainingType::NewTraining);
    Layer *l6 = new ConstructLayer(p_list6);
    l6->SetPrevLayer(l1);
    l6->Init();

    // 5) output layer ID = 7
    LayerParameterList p_list_output(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), 0.1, false, 0, 0., Regularization::L2, 0.1, ActuationFuncType::SoftMax, TrainingType::NewTraining);
    Layer* layer_output = new ConstructLayer(p_list_output);
    layer_output -> SetPrevLayer(l6);
    layer_output -> Init();

    // 6) connect all layers; SetNextLayer must be after all layers have finished initialization
    //                        input layer not needed to set, input layer has no update on w & b
    //                        input layer is just for data transfer (prepare)
    l1->SetNextLayer(l6);
    //l2->SetNextLayer(l3);
    l6->SetNextLayer(layer_output); // This line is ugly, to be improved


    // step 2: forward propagate
    data_interface -> Reset();
    layer_input->EpochInit();
    l1->EpochInit();
    l6->EpochInit();
    layer_output->EpochInit();

    //layer_input->BatchInit();
    l1->BatchInit();
    l6->BatchInit();
    layer_output->BatchInit();


    data_interface->GetNewBatchData();
    data_interface->GetNewBatchLabel();
    layer_input -> FillBatchDataToInputLayerA();


    // helper functions
    auto print_a_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'A' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullA();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };
    auto print_z_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'Z' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullZ();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };


    auto print_W_B_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::W&B matrix from layer: "<<l->GetID()<<endl;
	auto w_images = l->GetWeightMatrixOriginal();
	auto b_images = l->GetBiasVectorOriginal();
	assert((*w_images).size() == (*b_images).size());

	for(size_t kernel = 0;kernel<(*w_images).size();kernel++)
	{
	    cout<<"----kernel id: "<<kernel<<endl;
	    cout<<"w"<<kernel<<":"<<endl;
	    cout<<(*w_images)[kernel]<<endl;
	    cout<<"b"<<kernel<<":"<<endl;
	    cout<<(*b_images)[kernel]<<endl;
	}
    };

    // forward propagation
    int sample_size = data_interface->GetBatchSize();
    for(int i=0;i<sample_size;i++)
    {
	l1->ForwardPropagate_Z_ForSample(i);
	l1->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
	
	l6->ForwardPropagate_Z_ForSample(i);
	l6->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
	
	layer_output->ForwardPropagate_Z_ForSample(i);
	layer_output->ForwardPropagate_SigmaPrimeAndA_ForSample(i);
    }

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_a_images(l1);
    print_z_images(l1);
    //print_a_images(l6);
    //print_z_images(l6);


    // backward propagation
    for(int i=0;i<sample_size;i++)
    {
	// must inverse order
	layer_output->BackwardPropagateForSample(i);
	l6->BackwardPropagateForSample(i);
	l1->BackwardPropagateForSample(i);
    }

    auto print_delta_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'Delta' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullDelta();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };

    auto print_sigmaPrime_images = [&](Layer *l)
    {
	// print a images
	cout<<"XXXXXX::'SigmaPrime' images from layer: "<<l->GetID()<<endl;
	auto & images = l->GetImagesFullSigmaPrime();
	int sample_IIID = 0;
	for(auto &i: images)
	{
	    cout<<"---------------------- sample id: "<<sample_IIID<<endl;
	    int kernel_id = 0;
	    for(auto &j: i.OutputImageFromKernel)
	    {
		cout<<"----kernel id: "<<kernel_id<<endl;
		cout<<j<<endl;
		kernel_id++;
	    }
	    sample_IIID++;
	}
    };

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_delta_images(l6);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_W_B_images(l6); 
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_a_images(l6);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    //print_delta_images(l6);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_sigmaPrime_images(l1);
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    print_delta_images(l1);

}




void UnitTest::TestCNNWeightAndBiasEvolution()
{
    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    DataInterface *data_interface = new DataInterface("simulation_data/data_signal_train.dat", "simulation_data/data_cosmic_train.dat", LayerDimension::_2D, std::pair<int, int>(10, 10), 200);

    TrainingType resume_or_new_training = TrainingType::NewTraining;
    double learning_rate = 0.06;
    double regularization_factor = 0.01;

    // 3) input layer   ID=0
    LayerParameterList p_list0(LayerType::input, LayerDimension::_2D, data_interface, 0, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0, Regularization::Undefined, 0, ActuationFuncType::Undefined, resume_or_new_training);
    Layer* layer_input = new ConstructLayer(p_list0);
    layer_input->Init();

    // 4) middle layer 3 : cnn layer ID=1
    LayerParameterList p_list1(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 1, 
	    std::pair<size_t, size_t>(4, 4), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training);
    Layer *l1 = new ConstructLayer(p_list1);
    l1->SetPrevLayer(layer_input);
    l1->Init();

    // 4) middle layer 3 : cnn layer ID=3
    LayerParameterList p_list3(LayerType::cnn, LayerDimension::_2D, data_interface, 0, 1, 
	    std::pair<size_t, size_t>(4, 4), learning_rate, false, 0, 0.5, Regularization::L2, regularization_factor, ActuationFuncType::Relu, resume_or_new_training);
    Layer *l3 = new ConstructLayer(p_list3);
    l3->SetPrevLayer(l1);
    l3->Init();

    // 5) output layer ID = 6
    LayerParameterList p_list6(LayerType::output, LayerDimension::_1D, data_interface, 2, 0, 
	    std::pair<size_t, size_t>(0, 0), learning_rate, false, 0, 0., Regularization::L2, regularization_factor, ActuationFuncType::SoftMax, resume_or_new_training);
    Layer* layer_output = new ConstructLayer(p_list6);
    layer_output -> SetPrevLayer(l3);
    layer_output -> Init();

    l1->SetNextLayer(l3);
    l3->SetNextLayer(layer_output); // This line is ugly, to be improved

    cout<<"ooooooooo000000000oooooooooo000000000ooooooooooo00000000oooooooooo0000000000000ooooooooooooo"<<endl;
    int numEpoch = 50;
    for(int e=0;e<numEpoch;e++)
    {
        cout<<"...............Epoch: "<<e<<endl;
        data_interface -> Reset();
	data_interface -> Shuffle();

	int nBatch = 1;
	for(int i = 0; i< nBatch;i++)
	{
	    l1 -> BatchInit();
	    l3 -> BatchInit();
	    layer_output -> BatchInit();

	    data_interface -> GetNewBatchData();
	    data_interface -> GetNewBatchLabel();

	    layer_input -> FillBatchDataToInputLayerA();

	    int nSample = data_interface -> GetBatchSize();

	    // forward propagation
	    for(int sample_index = 0; sample_index < nSample;sample_index++)
	    {
	        l1 -> ForwardPropagate_Z_ForSample(sample_index);
	        l1 -> ForwardPropagate_SigmaPrimeAndA_ForSample(sample_index);
	
	        l3 -> ForwardPropagate_Z_ForSample(sample_index);
	        l3 -> ForwardPropagate_SigmaPrimeAndA_ForSample(sample_index);
	
	        layer_output -> ForwardPropagate_Z_ForSample(sample_index);
	        layer_output -> ForwardPropagate_SigmaPrimeAndA_ForSample(sample_index);
	    }

	    // backward propagation
	    for(int sample_index = 0; sample_index < nSample;sample_index++)
	    {
		layer_output -> BackwardPropagateForSample(sample_index);
		l3 -> BackwardPropagateForSample(sample_index);
		l1 -> BackwardPropagateForSample(sample_index);
	    }

	    // update weights and bias
	    l1->UpdateWeightsAndBias();
	    l3->UpdateWeightsAndBias();
	    layer_output->UpdateWeightsAndBias();
	}
    }
}

























