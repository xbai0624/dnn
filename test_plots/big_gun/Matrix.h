/*
 * This is a minimal functional library for matrix operations, 
 * designed for Convolutional Neural Networks.
 *
 * The function includes:
 *   -Normal operations:
 *       Matrix multiplication: C = A*B
 *       Matrix addition: C = A + B
 *       Matrix subtraction: C = A - B
 *   -Hadamard operations:
 *       Matrix hadamard multiplication: C = A^B
 *       Matrix hadamard division: C = A/B ; although this one won't be needed in CNN
 *   -Matrix and scalar operations:
 *       multiplication by a scalar: A = A*s
 *       division by a scalar: A = A/s
 *       addition by a scalar: A = A + s;
 *       subtraction: A = A - s;
 *   -Matrix re-shape:
 *       (pxq) -> (mxn)
 *   -Matrix Padding
 *   -Matrix GetSectioin
 *
 *   -Element-wise operation with an exteranl functor:
 *       A(functor); which means apply functor() to each of its elements
 *   -And also several convolution and correlation functions
 *
 *   For fast operation, you can turn on multithreading, how many threads
 *   can be configured in Matrix.cpp source file by a macro
 *
 *   currently only accept float type element
 *
 *   @Xinzhan Bai, v0.0 1-31-2019
 */

#ifndef __MATRIX_H
#define __MATRIX_H

#include <vector>
#include <ostream>

enum class HadamardOps 
{
    // Hadamard operations 
    plus, subtract, multiply, divide
};

class Matrix 
{
public:
    Matrix();
    Matrix(size_t, size_t);
    Matrix(std::pair<size_t, size_t>);
    Matrix(size_t, size_t, float); // initialize all elements with a float 
    Matrix(std::pair<size_t, size_t>, float); // initialize all elements with a float
    Matrix(std::vector<std::vector<float>> &); // initialize matrix with a 2D vector
    ~Matrix();

    // overload operator
    Matrix operator*(Matrix &); // mutiply

    Matrix operator+(Matrix &); // sum
    Matrix operator-(Matrix &); // sub
    Matrix operator^(Matrix &); // hadamard multiply
    Matrix operator/(Matrix &); // hadamard division
    Matrix operator+(float); // sum by a float
    Matrix operator-(float); // minus by a float
    Matrix operator*(float); // multiply by a float
    Matrix operator/(float); // divide by a float

    bool operator==(Matrix &); // matrix equal to each other

    std::vector<float>& operator[](size_t);
    size_t size() const;
    void operator()(float(*funtor)(float)); // apply a functor to matrix element, matrix itself will be changed
    void operator()(float(*funtor)(float), size_t r1, size_t r2, size_t c1, size_t c2); // same as above, but only apply to a section of the matrix

    // multiply
    void _mul_Multiply(Matrix &B, Matrix &C); // (*this) X B = C
    void _mul_ComputeSection(Matrix &B, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2);
    float _mul_ComputeElement(Matrix &B, size_t r, size_t c);
    // hadamard
    void _hadamard_ComputeSection(HadamardOps op, Matrix &B, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2);
    // scalar operation
    void _scalar_ComputeSection(HadamardOps op, const float v, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2);

    // set dimension
    void SetInitialDimension(size_t, size_t);

    // reshape
    Matrix Reshape(size_t m, size_t n) const;
    // transpose
    Matrix Transpose();
    // rotate
    Matrix Rot_180();
    // get a section of matrix [i, j) [m, n): (front close, back open)
    Matrix GetSection(size_t, size_t, size_t, size_t, bool padding=false, float padding_value=0);
    // sum all elements
    float ElementSum();
    // max element in section
    float MaxInSection(size_t, size_t, size_t, size_t);
    float MaxInSection(size_t, size_t, size_t, size_t, std::pair<size_t, size_t> &coord);
    float MaxInSectionWithPadding(size_t, size_t, size_t, size_t, float padding_value = -9.e10);
    // average in section
    float AverageInSection(size_t, size_t, size_t, size_t);
    float AverageInSectionWithPadding(size_t, size_t, size_t, size_t);
    // average in section
    float SumInSection(size_t, size_t, size_t, size_t);
    float SumInSectionWithPadding(size_t, size_t, size_t, size_t);
    // padding matrix
    Matrix Padding(size_t, size_t, bool pad_front=true, float padding_value=0); // padding matrix
    // correlation
    Matrix Correlation(Matrix &, size_t stride=1, bool pad_front=false, float padding_value=0);
    // convolution
    Matrix Convolution(Matrix &, size_t stride=1, bool pad_front=false, float padding_value=0);

    // fill matrix with random numbers
    void Random();
    void RandomGaus(float mu, float sigma);
    std::pair<size_t, size_t> Dimension() const;

    void Clear();
    // fill element, row first
    void FillElementByRow(size_t, float);
    // fill element, collum first
    void FillElementByCollum(size_t, float);
    // delete row
    void DeleteRow(size_t i);
    // delete collum
    void DeleteCollum(size_t j);
    // insert row
    void InsertRow(size_t, std::vector<float>* ptr=nullptr);
    // insert collum
    void InsertCollum(size_t, std::vector<float>* ptr=nullptr);

    //static members
    // combine multiple matrix (same dimension) to one big matrix
    static Matrix CombineMatrix(std::vector<Matrix>&, size_t, size_t);
    // Dispatch one matrix to multiple small matrix (same dimension)
    static std::vector<Matrix> DispatchMatrix(Matrix&, size_t, size_t);
    // concatenate two matrix in i direction / vertically
    static Matrix ConcatenateMatrixByI(Matrix &, Matrix &);
    // concatenate a vector of matrix in i direction / vertically
    static Matrix ConcatenateMatrixByI(std::vector<Matrix> &);
    // concatenate two matrix in j direction / horizontally
    static Matrix ConcatenateMatrixByJ(Matrix &, Matrix &);
    // concatenate a vector of matrix in j direction / horizontally
    static Matrix ConcatenateMatrixByJ(std::vector<Matrix> &);
    // Get element (i, j) of the correlated matrix, A is input, B is kernel
    static float GetCorrelationValue(Matrix &A, Matrix &B, size_t i, size_t j);
    // Get element (i, j) of the convoluted matrix, A is input, B is kernel
    static float GetConvolutionValue(Matrix &A, Matrix &B, size_t i, size_t j);

private:
    std::vector<std::vector<float>> __M;
};

std::ostream& operator<<(std::ostream& os, Matrix&);
std::ostream& operator<<(std::ostream& os, std::vector<float>&);
bool operator==(std::pair<size_t, size_t> left, std::pair<size_t, size_t> right);
bool operator!=(std::pair<size_t, size_t> left, std::pair<size_t, size_t> right);
std::ostream& operator<<(std::ostream &os, const std::pair<size_t, size_t> &d);

#endif
