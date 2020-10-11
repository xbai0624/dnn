#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <cassert>

bool operator==(std::pair<size_t, size_t> left, std::pair<size_t, size_t> right)
{
    // directly compare matrix dimension
    if(left.first == right.first && left.second == right.second)
        return true;
    return false;
}

bool operator!=(std::pair<size_t, size_t> left, std::pair<size_t, size_t> right)
{
    // directly compare matrix dimension
    if(left.first != right.first || left.second != right.second)
        return true;
    return false;
}

std::ostream &operator<<(std::ostream &os, const std::pair<size_t, size_t> &d)
{
    os<<std::setfill(' ')<<std::setw(6)<<d.first
        <<std::setfill(' ')<<std::setw(6)<<d.second
        <<std::endl;
    return os;
}

//#define MULTITHREAD_MATRIX
#define NCORES 4

#ifdef MULTITHREAD_MATRIX
static std::vector<std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>> DivideJobs(size_t rows, size_t cols, size_t cores)
{
    // how to dispatch jobs between different threads,
    // each threads get how many jobs
    // "cores" must be even numbers
    //
    // the division will be like:
    // | job_1 | job_2 | ... | job_n/2 |
    // _________________________________
    // | job_1 | job_2 | ... | job_n/2 |
    //
    // vertically 2 sections, horizontally cores/2 sections

    std::vector<std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>> res;

    size_t xw = rows/2;

    size_t yw = cols/(cores/2);

    for(size_t i=0;i<cores/2;i++){
        size_t y_end = (i+1)*yw;
        if( i == cores/2 -1 ) y_end = cols;
        res.emplace_back(std::make_pair<size_t, size_t>(0, 0+xw), std::make_pair<size_t, size_t>(i*yw, 0+y_end)); // top half; the "0+" part is to get rid of error information
        res.emplace_back(std::make_pair<size_t, size_t>(0+xw, 0+rows), std::make_pair<size_t, size_t>(i*yw, 0+y_end)); // bottom half
    }

    return res;
}
#endif

// method 1
//static std::random_device rd;
//std::atomic<unsigned int> Matrix::seed{rd()};

// method 2
//std::atomic<unsigned int> Matrix::seed{491856570}; // initialize seed for random generator

// method 3
std::atomic<unsigned int> Matrix::seed{1}; // initialize seed for random generator

Matrix::Matrix()
{
    // defulat constructor
    __M.clear();
    __M.resize(1, std::vector<double>(1, 0));
}

Matrix::Matrix(size_t r, size_t c) 
{
    // constructor
    __M.clear();
    __M.resize(r, std::vector<double>(c, 0));
}

Matrix::Matrix(std::pair<size_t, size_t> c)
{
    // constructor
    __M.clear();
    __M.resize(c.first, std::vector<double>(c.second, 0));
}

Matrix::Matrix(size_t r, size_t c, double v) 
{
    // constructor and initialize all elements with a value v
    __M.clear();
    __M.resize(r, std::vector<double>(c, v));
}

Matrix::Matrix(std::pair<size_t, size_t> c, double v)
{
    // constructor and initialize all elements with a value v
    __M.clear();
    __M.resize(c.first, std::vector<double>(c.second, v));
}

Matrix::Matrix(std::vector<std::vector<double>> &vv)
{
    // constructor and initilize matrix with a 2D vector vv
    size_t nRow = vv.size();
    for(size_t i=0;i<nRow;i++)
        __M.push_back(vv[i]);
}

Matrix::Matrix(const Matrix &m)
{
    // copy constructor
    auto dim = m.Dimension();
    __M.clear();
    __M.resize(dim.first, std::vector<double>(dim.second, 0));
    for(size_t i=0;i<dim.first;i++){
        for(size_t j=0;j<dim.second;j++)
        {
            __M[i][j] = m.GetElement(i, j);
        }
    }

    assert(Dimension() == dim); // make sure copy is correct
}

Matrix::~Matrix()
{
    // place holder
    __M.clear();
}

Matrix Matrix::operator*(Matrix &B)
{
    // matrix multiply operation *: A*B = C
    if(this->size() == 0 || B.size() == 0) {
        std::cout<<__func__<<" Error: 0 Matrix."<<std::endl;
        std::cout<<"    "<<this->size()<<" rows in left Matrix [times] "<<B.size()<<" rows in right Matrix."<<std::endl;
        exit(0);
    }

    if((*this)[0].size() != B.size() ) {
        std::cout<<__func__<<" Error: Matrix not match."<<std::endl;
        std::cout<<"    "<<(*this)[0].size()<<" collums in left Matrix [times] "<<B.size()<<" rows in right Matrix."<<std::endl;
        exit(0);
    }

    Matrix _r((*this).size(), B[0].size());

    _mul_Multiply(B, _r);

    return _r;
}

Matrix Matrix::operator+(Matrix &B)
{
    // matrix plus operation +: A+B = C
    if(B.Dimension() != this->Dimension()){
        std::cout<<"Error: Matrix not match in + ops."<<std::endl;
        exit(0);
    }

    Matrix C(B.Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), B[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, &B, &C, &i](){
                    _hadamard_ComputeSection(HadamardOps::plus, B, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _hadamard_ComputeSection(HadamardOps::plus, B, C, 0, B.Dimension().first, 0, B.Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator-(Matrix &B)
{
    // matrix plus operation -: A-B = C
    if(B.Dimension() != this->Dimension()){
        std::cout<<"Error: Matrix not match in - ops."<<std::endl;
        exit(0);
    }

    Matrix C(B.Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), B[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, &B, &C, &i](){
                    _hadamard_ComputeSection(HadamardOps::subtract, B, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _hadamard_ComputeSection(HadamardOps::subtract, B, C, 0, B.Dimension().first, 0, B.Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator^(Matrix &B)
{
    // matrix plus operation ^: A^B = C
    if(B.Dimension() != this->Dimension()){
        std::cout<<"Error: Matrix not match in ^ ops."<<std::endl;
        exit(0);
    }

    Matrix C(B.Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), B[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, &B, &C, &i](){
                    _hadamard_ComputeSection(HadamardOps::multiply, B, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _hadamard_ComputeSection(HadamardOps::multiply, B, C, 0, B.Dimension().first, 0, B.Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator/(Matrix &B)
{
    // matrix plus operation /: A/B = C
    if(B.Dimension() != this->Dimension()){
        std::cout<<"Error: Matrix not match in / ops."<<std::endl;
        exit(0);
    }

    Matrix C(B.Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), B[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, &B, &C, &i](){
                    _hadamard_ComputeSection(HadamardOps::divide, B, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _hadamard_ComputeSection(HadamardOps::divide, B, C, 0, B.Dimension().first, 0, B.Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator+(double v)
{
    // scalar plus operation: A+v = C
    Matrix C(this->Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), (*this)[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, v, &C, &i](){
                    _scalar_ComputeSection(HadamardOps::plus, v, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _scalar_ComputeSection(HadamardOps::plus, v, C, 0, (*this).Dimension().first, 0, (*this).Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator-(double v)
{
    // scalar subtraction operation: A-v = C
    Matrix C(this->Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), (*this)[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, v, &C, &i](){
                    _scalar_ComputeSection(HadamardOps::subtract, v, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _scalar_ComputeSection(HadamardOps::subtract, v, C, 0, (*this).Dimension().first, 0, (*this).Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator*(double v)
{
    // scalar multiply operation: A*v = C
    Matrix C(this->Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), (*this)[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, v, &C, &i](){
                    _scalar_ComputeSection(HadamardOps::multiply, v, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _scalar_ComputeSection(HadamardOps::multiply, v, C, 0, (*this).Dimension().first, 0, (*this).Dimension().second);
#endif
    return C;
}

Matrix Matrix::operator/(double v)
{
    // scalar division operation: A/v = C
    Matrix C(this->Dimension());
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), (*this)[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back( std::thread( [this, v, &C, &i](){
                    _scalar_ComputeSection(HadamardOps::divide, v, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _scalar_ComputeSection(HadamardOps::divide, v, C, 0, (*this).Dimension().first, 0, (*this).Dimension().second);
#endif
    return C;
}

void Matrix::_hadamard_ComputeSection(HadamardOps op, Matrix &B, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2)
{
    // compute a region inside C for hadamard operation
    for(size_t i=r1; i<r2; i++)
    {
        for(size_t j=c1; j<c2; j++){
            if(op == HadamardOps::plus)
                C[i][j] = (*this)[i][j] + B[i][j];
            else if (op == HadamardOps::subtract)
                C[i][j] = (*this)[i][j] - B[i][j];
            else if (op == HadamardOps::multiply)
                C[i][j] = (*this)[i][j] * B[i][j];
            else if (op == HadamardOps::divide)
                C[i][j] = (*this)[i][j] / B[i][j];
            else
                std::cout<<"Error: not supported hadamard operation."<<std::endl;
        }
    }
}

void Matrix::_scalar_ComputeSection(HadamardOps op, const double v, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2)
{
    // compute a region inside C for scalar operation
    for(size_t i=r1; i<r2; i++)
    {
        for(size_t j=c1; j<c2; j++){
            if(op == HadamardOps::plus)
                C[i][j] = (*this)[i][j] + v;
            else if (op == HadamardOps::subtract)
                C[i][j] = (*this)[i][j] - v;
            else if (op == HadamardOps::multiply)
                C[i][j] = (*this)[i][j] * v;
            else if (op == HadamardOps::divide)
                if( v != 0)
                    C[i][j] = (*this)[i][j] / v;
                else {
                    std::cout<<"Error: dividing a Matrix by 0"<<std::endl;
                }
            else
                std::cout<<"Error: not supported hadamard operation."<<std::endl;
        }
    }
}

std::vector<double>& Matrix::operator[](size_t i)
{
    // implement a 2d array like operation
    return __M[i];
}

double Matrix::GetElement(size_t i, size_t j) const
{
    // a helper, for copy constructor
    return __M[i][j];
}

size_t Matrix::size() const 
{
    // implement a 2d array like operation
    return __M.size();
}

void Matrix::_mul_Multiply(Matrix &B, Matrix &C)
{
    // AxB = C; A is this matrix
    if(C.size() != this->size() || C[0].size() != B[0].size()) {
        std::cout<<"Error: Matrix not match."<<std::endl;
        exit(0);
    }
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), B[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        //std::cout<<"("<<i.first.first<<", "<<i.first.second<<"), ";
        //std::cout<<"("<<i.second.first<<", "<<i.second.second<<"), "<<std::endl;
        th.push_back( std::thread( [this, &B, &C, &i](){
                    _mul_ComputeSection(B, C, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    ) 
                );
    }
    for(auto &j: th){
        j.join();
    }
#else
    _mul_ComputeSection(B, C, 0, this->size(), 0, B[0].size());
#endif
}

void Matrix::_mul_ComputeSection(Matrix &B, Matrix &C, size_t r1, size_t r2, size_t c1, size_t c2)
{
    // compute a section of C
    if(r1 >= r2 || c1 >= c2) {
        return;
    }
    if( (int)r1<0 || r2>this->size() || (int)c1<0 || c2>B[0].size()) {
        std::cout<<"Error: matrix out of range."<<std::endl;
        exit(0);
    }
    if(C.size() != this->size() || C[0].size() != B[0].size() ) {
        std::cout<<"Error: matrix not match."<<std::endl;
        exit(0);
    }

    for(size_t i=r1; i<r2;i++){
        for(size_t j=c1;j<c2;j++){
            C[i][j] = _mul_ComputeElement(B, i, j);
        }
    }
}

double Matrix::_mul_ComputeElement(Matrix &B, size_t r, size_t c)
{
    // compute an element in the result matrix, 
    // B is multiplied to this matrix: AxB = C
    double s = 0;
    for(size_t i=0;i<(*this)[0].size();i++){
        s += (*this)[r][i] * B[i][c];
    }
    return s;
}

std::ostream& operator<<(std::ostream &os, std::vector<Matrix> &vM)
{
    for(auto &i: vM)
    {
        os<<i<<std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream &os, Matrix& A)
{
    // overload cout for Matrix
    for(size_t i=0;i<A.size();i++){
        os<<A[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream &os, std::vector<double>& A)
{
    // overload cout for vector
    for(size_t i=0;i<A.size();i++){
        os<<std::setfill(' ')<<std::setw(18)<<std::setprecision(6)<<A[i];
    }
    os<<std::endl;
    return os;
}

void Matrix::Random()
{
    // fill with random numbers
    // uniform (0,1) distribution
    //std::cout<<"Filling matrix using flat (0, 1) distribution"<<std::endl;
    std::random_device rd;
    std::mt19937 mt(rd());
    //unsigned int SEED = seed;
    //std::mt19937 mt(SEED);
    //seed = seed + 1;
    std::uniform_real_distribution<double> dist(0, 1.);

    for(size_t i=0;i<__M.size();i++){
        for(size_t j=0;j<__M[0].size();j++){
            __M[i][j] = dist(mt);
        }
    }
    //std::cout<<"Fill finished."<<std::endl;
}

void Matrix::Random(double min, double max)
{
    // fill with random numbers
    // uniform (0,1) distribution
    //std::cout<<"Filling matrix using flat (0, 1) distribution"<<std::endl;
    std::random_device rd;
    std::mt19937 mt(rd());
    //unsigned int SEED = seed;
    //std::mt19937 mt(SEED);
    //seed = seed+1;
    std::uniform_real_distribution<double> dist(min, max);

    for(size_t i=0;i<__M.size();i++){
        for(size_t j=0;j<__M[0].size();j++){
            __M[i][j] = dist(mt);
        }
    }
    //std::cout<<"Fill finished."<<std::endl;
}

void Matrix::RandomGaus(double mu, double sigma)
{
    // fill with random numbers
    // uniform (0,1) distribution
    //std::cout<<"Filling matrix using flat (0, 1) distribution"<<std::endl;
    //std::random_device rd;
    //std::mt19937 mt(rd());

    unsigned int SEED = seed;
    std::mt19937 mt(SEED);
    seed = seed + 1;

    std::normal_distribution<double> dist(mu, sigma);
    for(size_t i=0;i<__M.size();i++){
        for(size_t j=0;j<__M[0].size();j++){
            __M[i][j] = dist(mt);
        }
    }
    //std::cout<<"Fill finished."<<std::endl;
}


std::pair<size_t, size_t> Matrix::Dimension() const
{
    // return matrix dimension
    return std::make_pair<size_t, size_t>(__M.size(), __M[0].size());
}

void Matrix::operator()(double(*functor)(double))
{
    // pass a functor to this matrix, and apply this functor to all its elements
#ifdef MULTITHREAD_MATRIX
    auto jobs = DivideJobs(this->size(), (*this)[0].size(), NCORES);
    std::vector<std::thread> th;
    for(auto &i: jobs)
    {
        th.push_back(std::thread( [this, &i, &functor]() {
                    (*this)(functor, i.first.first, i.first.second, i.second.first, i.second.second);
                    }
                    )
                );
    }
    for(auto &i: th)
        i.join();
#else
    (*this)(functor, 0, this->size(), 0, (*this)[0].size());
#endif
}

void Matrix::operator()(double(*functor)(double), size_t r1, size_t r2, size_t c1, size_t c2)
{
    // pass a functor, apply to a region of the matrix
    if( (int)r1<0 || r2>(*this).size() || r2 <= r1 || c2 <= c1
            || (int)c1<0 || c2 > (*this)[0].size()
      ) 
        return;

    for(size_t i=r1;i<r2;i++){
        for(size_t j=c1;j<c2;j++){
            (*this)[i][j] = (*functor)( (*this)[i][j] );
        }
    }
}

bool Matrix::operator==(Matrix &m)
{
    auto dim = m.Dimension();
    if(Dimension() != dim) return false;

    for(size_t i=0;i<dim.first;i++)
        for(size_t j=0;j<dim.second;j++)
        {
            //if(m[i][j] != (*this)[i][j]) return false;
            if(abs(m[i][j] - (*this)[i][j]) > 1.e-6) return false;
        }

    return true;
}

Matrix Matrix::Reshape(size_t m, size_t n) const
{
    // reshape matrix from original (p, q) to (m, n)
    size_t p = this->size();
    size_t q = (*(const_cast<Matrix*>(this)))[0].size();
    size_t T = p*q; // total # of elements

    if(m==0 || n==0 || p==0 || q==0 || T != m*n) {
        std::cout<<"Error: Reshape matrix dimension not match"<<std::endl;
        exit(0);
    }

    Matrix res(m, n);
    for(size_t i=0;i<T;i++){
        size_t new_x = i/n, new_y = i%n;
        size_t old_x = i/q, old_y = i%q;

        res[new_x][new_y] = (*(const_cast<Matrix*>(this)))[old_x][old_y];
    }

    return res;
}

Matrix Matrix::Normalization()
{
    // includes normalization, centering and standarization per sample
    auto dim = Dimension();
    Matrix m(dim);

    // step 1) normalize
    double min = MinInSection(0, dim.first, 0, dim.second);
    double max = MaxInSection(0, dim.first, 0, dim.second);

    double amp = max - min;

    if(amp > 0)
        for(size_t i=0;i<dim.first;i++)
            for(size_t j=0;j<dim.second;j++)
                m[i][j] = (*this)[i][j] / amp;

    // step 2) cenering
    double average = 0;
    // find min, max, mean
    for(size_t i=0;i<dim.first;i++){
        for(size_t j=0;j<dim.second;j++)
        {
            average += m[i][j];
        }
    }
    double N = dim.first * dim.second;
    average /= N;
    if(average != 0)
        for(size_t i=0;i<dim.first;i++){
            for(size_t j=0;j<dim.second;j++)
            {
                m[i][j] = m[i][j] - average;
            }
        }

    // step 3) standardization
    // find sigma
    double sigma = 0;
    for(size_t i=0;i<dim.first;i++){
        for(size_t j=0;j<dim.second;j++)
        {
            double tmp = m[i][j];
            sigma += tmp*tmp;
        }
    }
    sigma = sqrt(sigma);
    if(sigma > 0)
        for(size_t i=0;i<dim.first;i++){
            for(size_t j=0;j<dim.second;j++)
            {
                m[i][j]/=sigma;
            }
        }

    return m;
}

Matrix Matrix::Transpose()
{
    // transpose current matrix
    auto d = Dimension();
    Matrix _m(d.second, d.first);
    for(size_t i=0;i<d.first;i++){
        for(size_t j=0;j<d.second;j++){
            _m[j][i] = __M[i][j];
        }
    }
    return _m;
}

void Matrix::SetInitialDimension(size_t m, size_t n)
{
    // this is in order to work with the default empty constructor
    __M.clear();
    __M.resize(m, std::vector<double>(n, 0));
}

void Matrix::Clear()
{
    // clear matrix
    __M.clear();
}

void Matrix::FillElementByRow(size_t index, double v)
{
    // fill an element to this matrix, row first
    auto dim = Dimension();
    size_t i = index/dim.second;
    size_t j = index%dim.second;
    if(i >= dim.first || j>= dim.second) {
        std::cout<<"Error: filling matrix exceeded range."<<std::endl;
        exit(0);
    }
    __M[i][j] = v;
}

void Matrix::FillElementByCollum(size_t index, double v)
{
    // fill an element to this matrix, collum first
    auto dim = Dimension();
    size_t i = index%dim.first;
    size_t j = index/dim.first;
    if(i >= dim.first || j>= dim.second) {
        std::cout<<"Error: filling matrix exceeded range."<<std::endl;
        exit(0);
    }
    __M[i][j] = v;
}

double Matrix::ElementSum()
{
    // sum all elements together
    double res = 0;
    auto dim = Dimension();
    for(size_t i=0;i<dim.first;i++){
        for(size_t j=0;j<dim.second;j++){
            res += __M[i][j];
        }
    }
    return res;
}

Matrix Matrix::GetSection(size_t r_s, size_t r_e, size_t c_s, size_t c_e, 
        bool padding, double padding_value)
{
    // get a section of this matrix
    auto dim = Dimension();

    if( c_e <= c_s || r_e <= r_s || (int)c_s<0 || (int)r_s<0 || r_s>=dim.first || c_s>=dim.second ) 
    {
        std::cout<<"Error: wrong index provided in fetching matrix section."<<std::endl;
        std::cout<<"       front close, back open."<<std::endl;
        exit(0);
    }

    if(!padding) {
        if(r_e > dim.first || c_e > dim.second) {
            std::cout<<"Error: matrix fetching section exceeded range."<<std::endl;
            exit(0);
        }
    }

    Matrix _m(r_e - r_s, c_e - c_s);
    for(size_t i=0;i<(r_e-r_s);i++)
    {
        for(size_t j=0;j<(c_e-c_s);j++){
            if( (r_s + i) < dim.first && (c_s + j) < dim.second ){
                _m[i][j] = __M[r_s+i][c_s+j];
            } else {
                _m[i][j] = padding_value;
            }
        }
    }
    return _m;
}

Matrix Matrix::Padding(size_t ROW, size_t COL, bool pad_front, double padding_value)
{
    // matrix padding, only two modes supported: 1) only pad back, 2) pad back and front
    // only pad front not supported, b/c it's not useful reallistically
    auto dim = Dimension();
    if(ROW < dim.first || COL < dim.second) {
        std::cout<<"Error: trying pad matrix to a smaller size, no padding needed."<<std::endl;
        exit(0);
    }

    Matrix _m(ROW, COL, padding_value);
    if( ! pad_front){ // only pad back
        for(size_t i=0;i<dim.first;i++){
            for(size_t j=0;j<dim.second;j++){
                _m[i][j] = __M[i][j];
            }
        }
    } else { // pad back and front
        size_t extra_row = ROW - dim.first, extra_col = COL - dim.second;
        size_t extra_front_row = extra_row/2, extra_front_col = extra_col/2;
        for(size_t i=0;i<dim.first;i++){
            for(size_t j=0;j<dim.second;j++){
                _m[extra_front_row+i][extra_front_col+j] = __M[i][j];
            }
        }
    }

    return _m;
}

Matrix Matrix::CombineMatrix(std::vector<Matrix> &vec, size_t R, size_t C)
{
    // combine a vector of matrix'es to one big matrix,
    // the big matrix dimension is R, C
    Matrix _m(R, C);
    size_t N = vec.size();
    auto dim = vec[0].Dimension();
    if(R*C != N * dim.first * dim.second ) {
        std::cout<<"Error: combining matrix dimension not match."<<std::endl;
        exit(0);
    }

    size_t index = 0;
    for(auto &m: vec){
        for(size_t i=0;i<dim.first;i++){
            for(size_t j=0;j<dim.second;j++){
                _m.FillElementByRow(index, m[i][j]);
                index++;
            }
        }
    }

    if(index != R*C) {
        std::cout<<"Error: combining matrix dimension not match."<<std::endl;
        exit(0);
    }

    return _m;
}

std::vector<Matrix> Matrix::DispatchMatrix(Matrix & M, size_t R, size_t C)
{
    // dispatch one big matrix to multiple small matrix'es, 
    // return in vector form
    // small matrix dimension R, C
    std::vector<Matrix> res;
    auto dim = M.Dimension();
    if( (dim.first*dim.second) % (R*C) != 0) {
        std::cout<<"Error: dispatch matrix dimension not match."<<std::endl;
        exit(0);
    }

    size_t index = 0;
    Matrix tmp(R, C);

    for(size_t i=0;i<dim.first;i++)
    {
        for(size_t j=0;j<dim.second;j++){
            tmp.FillElementByRow(index, M[i][j]);
            index++;
            if(index == R*C) {
                res.push_back(tmp);
                index = 0;
            }
        }
    }
    return res;
}

Matrix Matrix::ConcatenateMatrixByI(Matrix &A, Matrix &B)
{
    // vertially place two matrix together, 
    auto dimA = A.Dimension();
    auto dimB = B.Dimension();
    if(dimA.second != dimB.second){
        std::cout<<"Error: place two matrix together vertically, dimension not match."<<std::endl;
        exit(0);
    }

    Matrix C(dimA.first + dimB.first, dimA.second);
    for(size_t i=0;i<dimA.first;i++){
        for(size_t j=0;j<dimA.second;j++){
            C[i][j] = A[i][j];
        }
    }
    for(size_t i=dimA.first;i<dimA.first+dimB.first;i++){
        for(size_t j=0;j<dimB.second;j++){
            C[i][j] = B[i-dimA.first][j];
        }
    }
    return C;
}


Matrix Matrix::ConcatenateMatrixByI(std::vector<Matrix> &A)
{
    // vertically place a vector of matrix together, 

    // method 1: seems pretty slow
    //Matrix C = A[0];

    //for(size_t i=1;i<A.size();i++)
    //{
    //    auto dimAI = A[i].Dimension();
    //    for(size_t ii=0;ii<dimAI.first;ii++)
    //    {
    //        auto dimC = C.Dimension();
    //        assert(dimC.second == dimAI.second);
    //        C.InsertRow(dimC.first, &A[i][ii]);
    //    }
    //}

    // method 2
    auto dim = A[0].Dimension();
    size_t length = A.size();

    Matrix C(dim.first*length, dim.second, 0);

    for(size_t i=0;i<length;i++)
    {
        assert(dim.second == A[i].Dimension().second);
        assert(dim.first == A[i].Dimension().first);

        for(size_t ii=0;ii<dim.first;ii++)
            for(size_t jj=0;jj<dim.second;jj++)
                C[i*dim.first + ii][jj] = (A[i])[ii][jj];
    }

    return C;
}

Matrix Matrix::ConcatenateMatrixByI(std::vector<Matrix> *A)
{
    // vertically place a vector of matrix together, 
    // method 2
    auto dim = (*A)[0].Dimension();
    size_t length = A->size();

    Matrix C(dim.first*length, dim.second, 0);

    for(size_t i=0;i<length;i++)
    {
        assert(dim.second == (*A)[i].Dimension().second);
        assert(dim.first == (*A)[i].Dimension().first);

        for(size_t ii=0;ii<dim.first;ii++)
            for(size_t jj=0;jj<dim.second;jj++)
                C[i*dim.first + ii][jj] = ((*A)[i])[ii][jj];
    }

    return C;
}



Matrix Matrix::ConcatenateMatrixByJ(Matrix &A, Matrix &B)
{
    // horizontally place two matrix together, 
    auto dimA = A.Dimension();
    auto dimB = B.Dimension();
    if(dimA.first != dimB.first){
        std::cout<<"Error: place two matrix together horizontally, dimension not match."<<std::endl;
        exit(0);
    }

    Matrix C(dimA.first, dimA.second + dimB.second);
    for(size_t j=0;j<dimA.second;j++){
        for(size_t i=0;i<dimA.first;i++){
            C[i][j] = A[i][j];
        }
    }
    for(size_t j=dimA.second;j<dimA.second+dimB.second;j++){
        for(size_t i=0;i<dimB.first;i++){
            C[i][j] = B[i][j-dimA.second];
        }
    }
    return C;
}


Matrix Matrix::ConcatenateMatrixByJ(std::vector<Matrix> &A)
{
    // vertially place a vector of matrix together, 
    Matrix C = A[0];

    // to be implemented

    return C;
}

double Matrix::MinInSection(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
{
    // find max element in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    if((int)i_start < 0 || (int)j_start < 0 || i_end <= i_start || j_end <= j_start || i_end > dim.first || j_end > dim.second){
        std::cout<<"Error: Min in matrix section: exceeded range."<<std::endl;
        exit(0);
    }
    double res = __M[i_start][j_start];
    for(size_t i=i_start;i<i_end;i++){
        for(size_t j=j_start;j<j_end;j++){
            if(res > __M[i][j])
                res = __M[i][j];
        }
    }
    return res;
}


double Matrix::MaxInSection(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
{
    // find max element in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    if((int)i_start < 0 || (int)j_start < 0 || i_end <= i_start || j_end <= j_start || i_end > dim.first || j_end > dim.second){
        std::cout<<"Error: Max in matrix section: exceeded range."<<std::endl;
        exit(0);
    }
    double res = __M[i_start][j_start];
    for(size_t i=i_start;i<i_end;i++){
        for(size_t j=j_start;j<j_end;j++){
            if(res < __M[i][j])
                res = __M[i][j];
        }
    }
    return res;
}


double Matrix::MaxInSection(size_t i_start, size_t i_end, size_t j_start, size_t j_end, std::pair<size_t, size_t> &coord)
{
    // find max element in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    if((int)i_start < 0 || (int)j_start < 0 || i_end <= i_start || j_end <= j_start || i_end > dim.first || j_end > dim.second){
        std::cout<<"Error: Max in matrix section: exceeded range."<<std::endl;
        exit(0);
    }
    double res = __M[i_start][j_start];
    for(size_t i=i_start;i<i_end;i++){
        for(size_t j=j_start;j<j_end;j++)
        {
            if(res < __M[i][j]){
                res = __M[i][j];
                coord.first = i;
                coord.second = j;
            }
        }
    }
    return res;
}


double Matrix::MaxInSectionWithPadding(size_t i_start, size_t i_end, size_t j_start, size_t j_end, double padding_value)
{
    // find max element in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    //
    // if range exceeded matirx dimension, then use padding value 
    auto dim = Dimension();
    bool at_least_one_element_in_range = false;

    assert(dim.first > 0 && dim.second > 0);
    assert((int)i_start >= 0 && i_end > i_start);
    assert((int)j_start >= 0 && j_end > j_start);

    auto in_range = [&](size_t ii, size_t jj) -> bool
    {
        if((int)ii>=0 && ii<dim.first && (int)jj>=0 && jj<dim.second)
            return true;
        return false;
    };

    double res = 0.;
    if(in_range(i_start, j_start))
        res = __M[i_start][j_start];
    else
        res = padding_value;

    for(size_t i=i_start;i<i_end;i++) {
        for(size_t j=j_start;j<j_end;j++)
        {
            if(! in_range(i, j)) continue;
            at_least_one_element_in_range = true;

            if(res < __M[i][j])
                res = __M[i][j];
        }
    }
    if(!at_least_one_element_in_range)
    {
        std::cout<<__func__<<"() Error: all elements exceeded range."<<std::endl;
        exit(0);
    }

    return res;
}


double Matrix::SumInSection(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
{
    // find element sum in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    if((int)i_start < 0 || (int)j_start < 0 || i_end <= i_start || j_end <= j_start || i_end > dim.first || j_end > dim.second){
        std::cout<<"Error: Sum in matrix section: exceeded range."<<std::endl;
        exit(0);
    }
    double res =  0;
    for(size_t i=i_start;i<i_end;i++){
        for(size_t j=j_start;j<j_end;j++){
            res += __M[i][j];
        }
    }

    return res;
}

double Matrix::SumInSectionWithPadding(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
{
    // find element sum in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    bool one_hit = false;

    assert((int)i_start >= 0 && i_end > i_start);
    assert((int)j_start >= 0 && j_end > j_start);

    auto in_range = [&](size_t ii, size_t jj) -> bool
    {
        if((int)ii>=0 && ii<dim.first && (int)jj>=0 && jj<dim.second)
            return true;
        return false;
    };

    double res =  0;
    for(size_t i=i_start;i<i_end;i++){
        for(size_t j=j_start;j<j_end;j++)
        {
            if(! in_range(i, j)) continue;
            one_hit = true;

            res += __M[i][j];
        }
    }

    if(!one_hit)
    {
        std::cout<<"Error: Sum in matrix section with padding: all elements exceeded range."<<std::endl;
        exit(0);
    }

    return res;
}


double Matrix::AverageInSection(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
{
    // find element average in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    if((int)i_start < 0 || (int)j_start < 0 || i_end <= i_start || j_end <= j_start || i_end > dim.first || j_end > dim.second){
        std::cout<<"Error: Max in matrix section: exceeded range."<<std::endl;
        exit(0);
    }
    double res =  0;
    for(size_t i=i_start;i<i_end;i++){
        for(size_t j=j_start;j<j_end;j++){
            res += __M[i][j];
        }
    }

    res = res /(int)(i_end-i_start)/(int)(j_end - j_start); 
    return res;
}


double Matrix::AverageInSectionWithPadding(size_t i_start, size_t i_end, size_t j_start, size_t j_end)
{
    // find element average in section [i_start, i_end), and [j_start, j_end)
    // in this code, we all follow the rule: close front and open end
    // and all counters start from 0
    auto dim = Dimension();
    bool one_hit = false;

    assert((int)i_start >= 0 && i_end > i_start);
    assert((int)j_start >= 0 && j_end > j_start);

    auto in_range = [&](size_t ii, size_t jj) -> bool
    {
        if((int)ii>=0 && ii<dim.first && (int)jj>=0 && jj<dim.second)
            return true;
        return false;
    };

    size_t x_range = i_end < dim.first ? i_end : dim.first;
    size_t y_range = j_end < dim.second? j_end : dim.second;

    double res =  0;
    for(size_t i=i_start;i<x_range;i++){
        for(size_t j=j_start;j<y_range;j++)
        {
            if(!in_range(i, j)) continue;
            one_hit = true;

            res += __M[i][j];
        }
    }

    if(!one_hit)
    {
        std::cout<<"Error: Average in matrix section with padding: all elements exceeded range."<<std::endl;
        exit(0);
    }


    res = res / (int)(x_range-i_start)/(int)(y_range - j_start); 
    return res;
}


Matrix Matrix::Rot_180()
{
    // rotate matrix by 180 degree
    auto dim = this->Dimension();
    Matrix _m(dim);

    for(size_t i=0;i<dim.first;i++){
        for(size_t j=0;j<dim.second;j++){
            _m[i][j] = (*this)[dim.first - i -1][dim.second - j-1];
        }
    }

    return _m;
}

Matrix Matrix::Correlation(Matrix &T, size_t stride, bool pad_front, double padding_value)
{
    // matrix correlation
    auto l_dim = Dimension();
    auto r_dim = T.Dimension();

    if(l_dim.first < r_dim.first || l_dim.second < r_dim.second) {
        std::cout<<"Error: matrix correlation dimension not match."<<std::endl;
        exit(0);
    }

    Matrix M = (*this); // left operand
    size_t row, col;
    row = (l_dim.first - r_dim.first + 1) / stride;
    col = (l_dim.second - r_dim.second + 1) / stride;

    if(stride > 1) { // check if padding needed
        row += 1; col+=1;
        size_t extra_row = row*stride - l_dim.first;
        size_t extra_col = col*stride - l_dim.second;

        if( extra_row > 0  || extra_col > 0) {
            M = (*this).Padding(l_dim.first + extra_row, l_dim.second+extra_col, pad_front, padding_value);
        }
    }

    l_dim = M.Dimension();

    Matrix res(row, col);
    for(size_t i=0;i<row;i++){
        for(size_t j=0;j<col;j++){
            Matrix l = M.GetSection(i*stride, i*stride + r_dim.first, j*stride, j*stride + r_dim.second);
            Matrix _t = l^T;
            auto __D = _t.Dimension();
            res[i][j] = _t.SumInSection(0, __D.first, 0, __D.second);
        }
    }
    return res;
}

Matrix Matrix::Convolution(Matrix &T, size_t stride, bool pad_front, double padding_value)
{
    // matrix convolution
    // convolution is correlation with T rotated 180 degree
    Matrix _T = T.Rot_180();
    Matrix r = Correlation(_T, stride, pad_front, padding_value);
    return r;
}

double Matrix::GetCorrelationValue(Matrix &A, Matrix &B, size_t i, size_t j)
{
    // get the element (i, j) of the correlated matrix, A is input matrix, B is kernel
    // keep in mind this one does not care about matrix dimension, it will fill with 0 value
    // for non-exist matrix elements
    if( (int)i<0 || (int)j<0) {
        std::cout<<"Error: wrong index in get correlation value."<<std::endl;
        exit(0);
    }

    auto dimA = A.Dimension();
    auto dimB = B.Dimension();

    double res = 0;
    for(size_t p = 0; p<dimB.first;p++)
    {
        for(size_t q=0; q<dimB.second;q++){
            int index_i_A = (int)i + (int)p;
            int index_j_A = (int)j + (int)q;

            if(index_i_A < 0 || index_i_A >= (int)dimA.first || index_j_A < 0 || index_j_A >= (int)dimA.second) {
                res += 0.; // 0*B[p][q];
            } else {
                res += A[index_i_A][index_j_A] * B[p][q];
            }
        }
    }
    return res;
}

double Matrix::GetConvolutionValue(Matrix &A, Matrix &B, size_t i, size_t j)
{
    // get the element (i, j) of the convoluted matrix, A is input matrix, B is kernel
    // keep in mind this one does not care about matrix dimension, it will fill with 0 value
    // for non-exist matrix elements
    if( (int)i<0 || (int)j<0) {
        std::cout<<"Error: wrong index in get convolution value."<<std::endl;
        exit(0);
    }

    auto dimA = A.Dimension();
    auto dimB = B.Dimension();

    double res = 0;
    for(size_t p = 0; p<dimB.first;p++)
    {
        for(size_t q=0; q<dimB.second;q++){
            int index_i_A = (int)i - (int)p;
            int index_j_A = (int)j - (int)q;

            if(index_i_A < 0 || index_i_A >= (int)dimA.first || index_j_A < 0 || index_j_A >= (int)dimA.second) {
                res += 0.; // 0*B[p][q];
            } else {
                res += A[index_i_A][index_j_A] * B[p][q];
            }
        }
    }
    return res;
}

//#define PARALLEL_NORMALIZATION 8
void Matrix::BatchNormalization(std::vector<Matrix> &vM)
{
    // if only one matrix in vM, return. Otherwise the whitening procedure will set every element to 0.
    if(vM.size() <= 1) return; 
    auto dim = vM[0].Dimension();

    double epsilon = 1e-8;
    double batchSize = static_cast<double>(vM.size());

    auto get_mean_and_sigma = [&](std::vector<Matrix> &_vM, size_t i, size_t j, double &mean, double &sigma)
    {
        assert((int)i >= 0 && i < dim.first && (int)j >=0 && j < dim.second);

        mean = 0; sigma = 0;
        for(auto &m: _vM) {
            mean += m[i][j];
        }
        mean /= batchSize;

        for(auto &m: _vM){
            sigma += (m[i][j] - mean) * (m[i][j] - mean);
        }
        sigma /= batchSize;
        if(batchSize > 1.) sigma *= (batchSize/(batchSize - 1.)); // get the un-biased variate
        sigma = sqrt(sigma);
        sigma += epsilon; // avoid divide by zero error
    };

    // batch normalization
#ifndef PARALLEL_NORMALIZATION
    for(size_t i=0;i<dim.first;i++){
        for(size_t j=0;j<dim.second;j++)
        {
            double mean = 0, sigma = 0;
            get_mean_and_sigma(vM, i, j, mean, sigma);
            for(auto &m: vM)
                m[i][j] = (m[i][j] - mean)/sigma;
        }
    }
#else
    //int nElements = dim.first * dim.second;
    //int interval = nElements / PARALLEL_NORMALIZATION;
    //size_t Range[PARALLEL_NORMALIZATION+1];

    // to be implemented as needed in later, not much gain for now
#endif
}

void Matrix::DeleteRow(size_t i)
{
    // delete a row
    auto dim = Dimension();
    if(i>= dim.first || (int)i<0) {
        std::cout<<"Warning: deleting matrix row, index out of range."<<std::endl;
        return;
    }

    __M.erase(__M.begin() + i);
}

void Matrix::DeleteCollum(size_t i)
{
    // delete a collum
    auto dim = Dimension();
    if((int)i<0 || i>= dim.second){
        std::cout<<"Warning: deleting matrix collum, index out of range."<<std::endl;
        return;
    }
    for(size_t j=0;j<dim.first;j++){
        __M[j].erase(__M[j].begin()+i);
    }
}

void Matrix::InsertRow(size_t i, std::vector<double>* vec)
{
    // insert vec to matrix at position i
    auto dim = Dimension();
    if(vec!=nullptr && vec->size() != dim.second){
        std::cout<<"Error: inserting row to matrix, dimension not match."<<std::endl;
        exit(0);
    }
    if((int)i<0 || i>dim.first){
        std::cout<<"Error: insert row to matrix, wrong position."<<std::endl;
        exit(0);
    }
    if(vec == nullptr){
        std::vector<double> v(dim.second, 0);
        __M.insert(__M.begin()+i, v);
    } else {
        __M.insert(__M.begin()+i, (*vec));
    }
}

void Matrix::InsertCollum(size_t i, std::vector<double>* vec)
{
    // insert vec to matrix at position i
    auto dim=Dimension();
    if(vec!=nullptr && vec->size() != dim.first){
        std::cout<<"Error: inserting col to matrix, dimension not match."<<std::endl;
        exit(0);
    }
    if((int)i<0 || i>dim.second){
        std::cout<<"Error: insert col to matrix, wrong position."<<std::endl;
        exit(0);
    }
    for(size_t ii=0;ii<dim.first;ii++){
        if(vec == nullptr){
            __M[ii].insert(__M[ii].begin() + i, 0);
        } else {
            __M[ii].insert(__M[ii].begin() + i, (*vec)[ii]);
        }
    }
}
