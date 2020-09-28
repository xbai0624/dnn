#ifndef UNIT_TESTING_H
#define UNIT_TESTING_H


// an example playground, for testing different units and also for examples

class UnitTest
{
public:
    UnitTest();
    ~UnitTest();

    void Test();

    void TestImagesStruct();
    void TestMatrix();

    void TestDNN();
    void TestCNN();
    void TestCNNWeightAndBiasEvolution();
    void TestCNNToPooling();
    void TestCNNToCNN();

    void TestFilter2D();

private:
    // reserved
};



#endif
