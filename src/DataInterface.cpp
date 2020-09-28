#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>

#include "DataInterface.h"
#include "Tools.h"

using namespace std;

DataInterface::DataInterface()
{
    // place holder
}

DataInterface::DataInterface(LayerDimension ld)
{
    // 1)
    __gLayerDimension = ld;
    // place holder
    // currently only accepts 2D image format
    test();
}


DataInterface::DataInterface(const char* p_signal, const char* p_cosmic, LayerDimension ld)
{
    // 1) first need to set layer dimension
    __gLayerDimension = ld;

    // 2) then load data to memory
    // for code development
    loadFile(p_signal, test_training_signal);
    loadFile(p_cosmic, test_training_cosmic);
}

DataInterface::DataInterface(const char* p_signal, const char* p_cosmic, LayerDimension ld, std::pair<int, int> dim, int batch_size)
{
    // 1) first need to set layer dimension
    __gLayerDimension = ld;

    // set data dimension
    gBatchSize = batch_size;
    __dataDimensionFromParameter.first = dim.first;
    __dataDimensionFromParameter.second = dim.second;

    // 2) then load data to memory
    // for code development
    loadFile(p_signal, test_training_signal);
    loadFile(p_cosmic, test_training_cosmic);

    //cout<<"data dimension from parameter: "<<__dataDimensionFromParameter<<endl;
}


DataInterface::~DataInterface()
{
    // place holder
}

int DataInterface::GetNumberOfBatches()
{
    // this line is to maliciously cause troulbe, what a ...
    assert(test_training_signal.size() == test_training_cosmic.size());
    int total_entries = test_training_signal.size();

    int res = total_entries/gBatchSize;
    if(total_entries%gBatchSize == 0) return res;
    else return res+1;
}

void DataInterface::test()
{
    // only one image
    Matrix image(std::pair<int, int>(10, 10), 1);
    //std::cout<<"input image: "<<std::endl<<image<<std::endl;

    Matrix image2 = image.Reshape(100, 1);
    //std::cout<<"input image: "<<std::endl<<image2<<std::endl;


    __data.push_back(image2);

    // label for this image
    Matrix label1(std::pair<int, int>(10, 1), 0);
    label1[0][0] = 1.;
    __label.push_back(label1);
}

std::vector<Matrix>& DataInterface::GetNewBatchData()
{
    // return data in matrix form
    UpdateBatch(__data, __label);

    return __data;
}


std::vector<Matrix>& DataInterface::GetNewBatchLabel()
{
    if(gLabelIndex + 1 != gDataIndex) // this is to make sure one access data first, then label
    {
        cout<<"Error: DataInterface data & label are not aligned."<<endl;
	exit(0);
    }
    gLabelIndex++;

    return __label;
}

std::vector<Images>& DataInterface::GetNewBatchDataImage()
{
    // return data in Images form
    
    // 1) update batch
    UpdateBatch(__data, __label);
    assert(__data.size() == __label.size());

    // 2) reform
    for(size_t i=0;i<__data.size();i++)
    {
        // reform it in Images format
        Images _image;
	_image.OutputImageFromKernel.push_back(__data[i]);
	__data_image.push_back(_image);

	Images _image_label;
	_image_label.OutputImageFromKernel.push_back(__label[i]);
	__label_image.push_back(_image_label);
    }

    return __data_image;
}

std::vector<Images>& DataInterface::GetNewBatchLabelImage()
{
    if(gLabelIndex + 1 != gDataIndex) // this is to make sure one access data first, then label
    {
        cout<<"Error: DataInterface data & label are not aligned."<<endl;
	exit(0);
    }
    gLabelIndex++;

    return __label_image;
}


void DataInterface::UpdateBatch(vector<Matrix> &data, vector<Matrix> &label)
{
    // clear previous batch
    data.clear();
    label.clear();

    size_t total_elements = __dataDimension.first*__dataDimension.second;
    assert( total_elements > 0); // make sure data dimension has been set

    //---------------------------------------------
    // prepare data
    int batch_size = gBatchSize / 2; // because signal + cosmic = gBatchSize
    int offset = gDataIndex * batch_size;
    assert( (size_t)(offset + batch_size) <= test_training_signal.size() ); // to avoid exceed range issue

    for(int i=0;i<batch_size;i++) // signal data
    {
        // data signal
        Matrix M;	
	if(__gLayerDimension == LayerDimension::_1D)
	{
	    // if input layer is 1D, then reshape images into a collum vector
	    M = test_training_signal[offset+i].Reshape(total_elements, 1);
	}
	else
	{
	    M = test_training_signal[offset+i];
	}
        data.push_back(M);

        // label
	Matrix signal_label_m(2, 1, 0); // signal label
	signal_label_m[0][0] = 1;
	label.push_back(signal_label_m);
    }
    for(int i=0;i<batch_size;i++) // cosmic data
    {
        // data cosmic
	Matrix M;
	if(__gLayerDimension == LayerDimension::_1D)
	{
	    M = test_training_cosmic[offset+i].Reshape(total_elements, 1);
	}
	else
	{
	    M = test_training_cosmic[offset+i];
	}
        data.push_back(M);

	// label
	Matrix cosmic_label_m(2, 1, 0); // cosmic label
	cosmic_label_m[1][0] = 1;
        label.push_back(cosmic_label_m);
    }

    // get ready for next batch
    gDataIndex++;
}

void DataInterface::Reset()
{
    // reset indicators, used in epoch loop
    // each epoch should start over from the beginning
    gDataIndex = 0;
    gLabelIndex = 0;
}

void DataInterface::Shuffle()
{
    TOOLS::Shuffle(test_training_signal);
    TOOLS::Shuffle(test_training_cosmic);
}

void DataInterface::loadFile(const char* path, std::vector<Matrix> &contents)
{
    // this only for code development, it reads data in ./test_data/ directory
    fstream f(path, fstream::in);
    if(!f.is_open()) 
    {
        std::cout<<__func__<<" Error: Cannot open file: "<<path<<std::endl;
	exit(0);
    }

    string line;
    while(getline(f, line))
    {
        istringstream iss(line);
	string tmp;
	vector<double> vec;
	while(iss>>tmp)
	{
	    if(tmp.size() > 0) 
	    {
	        double a = stod(tmp);
		vec.push_back(a);
	    }
	}
	assert(vec.size() == 27);

	size_t horizontal = 10;
	size_t vertical = 10;
	if(__dataDimensionFromParameter.first > 0)
	{
	    // if data dimension is set using the constructor parameter, then use it
	    horizontal = sqrt((int)__dataDimensionFromParameter.first * (int)__dataDimensionFromParameter.second);
	    vertical = horizontal;
	}
	Matrix m(horizontal, vertical); // 
	for(size_t i=0;i<vec.size();i+=3)
	{
	    size_t ii = vec[i];
	    size_t jj = vec[i+1];
	    assert(ii<horizontal && jj < vertical);
	    double val = vec[i+2];

	    m[ii][jj] = val;
	}

        Matrix _mm = m.Reshape(__dataDimensionFromParameter.first, __dataDimensionFromParameter.second);
        //Matrix mm = _mm.Normalization();
	contents.push_back(_mm);
    }

    // *** implement the image dimension
    assert(__gLayerDimension != LayerDimension::Undefined); // make sure 1D or 2D interface
    auto dim = contents[0].Dimension();
    if(__gLayerDimension == LayerDimension::_1D)
    {
	__dataDimension.second = 1;
	__dataDimension.first = dim.first * dim.second;
    }
    else 
    {
	__dataDimension = dim;
    }
}
