#ifndef DATA_INTERFACE_H
#define DATA_INTERFACE_H

/*
 *   This class is for helping input layer get data
 *          1) This class prepares data for input layer, it needs input layer Dimension information
 *             to prepare data with the correct dimension
 *             --) If input layer is 1D, then this class will prepare data in 1D form
 *             --) If input layer is 2D, then this class will prepare data in 2D form
 *
 *          2) Input layer is not a "normal" NN layer, it does not have neurons, 
 *          3) Input layer must have the __layerDimension parameter set: either 1D or 2D
 
 *          4) Currently, the LayerDimension info is only used in input_layer and DataInterface design
 *             other layers including output layer already have enough information to intialize
 */

#include "Matrix.h"
#include "Layer.h"

#include <vector>

class DataInterface
{
public:
    DataInterface(); // default constructor
    DataInterface(LayerDimension ld);
    DataInterface(const char* path1, const char* path2, LayerDimension ld); // for code development
    // for code development
    DataInterface(const char* path1, const char* path2, LayerDimension ld, std::pair<int, int> dim, int batch_size); 
    ~DataInterface();

    int GetBatchSize(){return gBatchSize;};
    void SetBatchSize(int s){gBatchSize = s;};
    int GetNumberOfBatches();

    // get out data in Matrix form
    std::vector<Matrix>& GetNewBatchData();
    std::vector<Matrix>& GetNewBatchLabel();
    std::vector<Matrix>& GetCurrentBatchData(){return __data;};
    std::vector<Matrix>& GetCurrentBatchLabel(){return __label;};

    // reform the data in Images form; 
    std::vector<Images>& GetNewBatchDataImage();
    std::vector<Images>& GetNewBatchLabelImage();
    std::vector<Images>& GetCurrentBatchDataImage(){return __data_image;};
    std::vector<Images>& GetCurrentBatchLabelImage(){return __label_image;};

    // a helper
    void UpdateBatch(std::vector<Matrix>& data_image, std::vector<Matrix>& label_image);
    void loadFile(const char* path, std::vector<Matrix> &m); // for code development

    // members
    std::pair<size_t, size_t> GetDataDimension(){return __dataDimension;};
    void Reset();
    void Shuffle();

    void test();

private:
    int gBatchSize = 100;
    int gDataIndex = 0; // indicate which batch
    int gLabelIndex = 0;

    // Get out batch data in Matrix form
    std::vector<Matrix> __data; // a vector of 2d image
    std::vector<Matrix> __label; // a vector of 2d image labels, used for training
    // Get out batch data in Images form
    std::vector<Images> __data_image; // a vector of 2d image
    std::vector<Images> __label_image; // a vector of 2d image labels, used for training

    std::pair<size_t, size_t> __dataDimension;
    std::pair<size_t, size_t> __dataDimensionFromParameter;

    //  Load all data to memory
    std::vector<Matrix> test_training_signal; // just for code development, loading all training data into this memory
    std::vector<Matrix> test_training_cosmic; // just for code development

    LayerDimension __gLayerDimension = LayerDimension::Undefined;
};

#endif
