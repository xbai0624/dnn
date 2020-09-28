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
 *   To make the program efficiency, currently only double type element is accepted
 *
 *   @Xinzhan Bai, v0.0 1-31-2019
 */

#ifndef __MATRIX_H
#define __MATRIX_H

#include <vector>
#include <ostream>
#include <atomic>

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
    Matrix(size_t, size_t, double); // initialize all elements with a double 
    Matrix(std::pair<size_t, size_t>, double); // initialize all elements with a double
    Matrix(std::vector<std::vector<double>> &); // initialize matrix with a 2D vector

    Matrix(const Matrix &); // copy constructor
    Matrix & operator=(const Matrix&) = default; // copy assignment
    //Matrix(Matrix &&) = default; // move constructor
    //Matrix& operator=(Matrix &&); // move assignment

    ~Matrix();

    // overload operator
    Matrix operator*(Matrix &); // mutiply

    Matrix operator+(Matrix &); // sum
    Matrix operator-(Matrix &); // sub
    Matrix operator^(Matrix &); // hadamard multiply
    Matrix operator/(Matrix &); // hadamard division
    Matrix operator+(double); // sum by a double
    Matrix operator-(double); // minus by a double
    Matrix operator*(double); // multiply by a double
    Matrix operator/(double); // divide by a double

    bool operator==(Matrix &); // matrix equal to each other

    std::vector<double>& operator[](size_t);
    size_t size() const;
    void operator()(double(*funtor)(double)); // apply a functor to matrix element, matrix itself will be changed
    void operator()(double(*funtor)(double), size_t r1, size_t r2, size_t c1, size_t c2); // same as above, but only apply to a section of the matrix
    double GetElement(size_t i, size_t j) const; // a helper for copy constructor

    // multiply
    void _mul_Multiply(Matrix &B, Matrix &C); // (*this) X B = C
    void _mul_ComputeSection(Matrix &B, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2);
    double _mul_ComputeElement(Matrix &B, size_t r, size_t c);
    // hadamard
    void _hadamard_ComputeSection(HadamardOps op, Matrix &B, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2);
    // scalar operation
    void _scalar_ComputeSection(HadamardOps op, const double v, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2);

    // set dimension
    void SetInitialDimension(size_t, size_t);

    // reshape
    Matrix Reshape(size_t m, size_t n) const;
    // transpose
    Matrix Transpose();
    // rotate
    Matrix Rot_180();
    // sample normalization (affine transformation over a single sample, potential use in Layer normalization algorithm)
    Matrix Normalization();
    // get a section of matrix [i, j) [m, n): (front close, back open)
    Matrix GetSection(size_t, size_t, size_t, size_t, bool padding=false, double padding_value=0);
    // sum all elements
    double ElementSum();
    // max element in section
    double MaxInSection(size_t, size_t, size_t, size_t);
    double MinInSection(size_t, size_t, size_t, size_t);
    double MaxInSection(size_t, size_t, size_t, size_t, std::pair<size_t, size_t> &coord);
    double MaxInSectionWithPadding(size_t, size_t, size_t, size_t, double padding_value = -9.e10);
    // average in section
    double AverageInSection(size_t, size_t, size_t, size_t);
    double AverageInSectionWithPadding(size_t, size_t, size_t, size_t);
    // average in section
    double SumInSection(size_t, size_t, size_t, size_t);
    double SumInSectionWithPadding(size_t, size_t, size_t, size_t);
    // padding matrix
    Matrix Padding(size_t, size_t, bool pad_front=true, double padding_value=0); // padding matrix
    // correlation
    Matrix Correlation(Matrix &, size_t stride=1, bool pad_front=false, double padding_value=0);
    // convolution
    Matrix Convolution(Matrix &, size_t stride=1, bool pad_front=false, double padding_value=0);

    // fill matrix with random numbers
    void Random();
    void Random(double min, double max);
    void RandomGaus(double mu, double sigma);
    std::pair<size_t, size_t> Dimension() const;

    void Clear();
    // fill element, row first
    void FillElementByRow(size_t, double);
    // fill element, collum first
    void FillElementByCollum(size_t, double);
    // delete row
    void DeleteRow(size_t i);
    // delete collum
    void DeleteCollum(size_t j);
    // insert row
    void InsertRow(size_t, std::vector<double>* ptr=nullptr);
    // insert collum
    void InsertCollum(size_t, std::vector<double>* ptr=nullptr);

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
    static double GetCorrelationValue(Matrix &A, Matrix &B, size_t A_i, size_t A_j);
    // Get element (i, j) of the convoluted matrix, A is input, B is kernel
    static double GetConvolutionValue(Matrix &A, Matrix &B, size_t A_i, size_t A_j);
    // Batch normalization for a set of matrices (aka whitening for input data, weighted sum, etc)
    static void BatchNormalization(std::vector<Matrix> &);

private:
    std::vector<std::vector<double>> __M;
    static std::atomic<unsigned int> seed; // seed for random generator
};

std::ostream& operator<<(std::ostream& os, Matrix&);
std::ostream& operator<<(std::ostream& os, std::vector<double>&);
std::ostream& operator<<(std::ostream& os, std::vector<Matrix>&);
bool operator==(std::pair<size_t, size_t> left, std::pair<size_t, size_t> right);
bool operator!=(std::pair<size_t, size_t> left, std::pair<size_t, size_t> right);
std::ostream& operator<<(std::ostream &os, const std::pair<size_t, size_t> &d);

#endif
