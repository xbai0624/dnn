#include <iostream>
#include <fstream>
#include <sstream>
#include "Matrix.h"

using namespace std;

vector<float> parseLine(string line)
{
    vector<float> res;
    istringstream iss(line);
    float v;
    while(iss>>v)
    {
	res.push_back(v);
    }
    return res;
}

vector<vector<float>> parseFile(const char* path)
{
    fstream ff(path, fstream::in);
    string line;
    vector<vector<float>> res;

    while(getline(ff, line))
    {
	vector<float> tmp = parseLine(line);
	res.push_back(tmp);
    }
    return res;
}




int main(int argc, char* argv[])
{
    auto _m1 = parseFile("m1.txt");
    auto _m2 = parseFile("m2.txt");


    Matrix m1(_m1);
    Matrix m2(_m2);

    Matrix R(4, 4, 0);

    for(size_t i=0;i<4;i++)
    for(size_t j=0;j<4;j++)
        R[i][j] = Matrix::GetConvolutionValue(m1, m2, i, j);

    //R[0][0] = Matrix::GetConvolutionValue(m1, m2, 0, 0);
    //R[0][1] = Matrix::GetConvolutionValue(m1, m2, 0, 1);
    //R[1][0] = Matrix::GetConvolutionValue(m1, m2, 1, 0);
    //R[1][1] = Matrix::GetConvolutionValue(m1, m2, 1, 1);
    //R[0][0] = Matrix::GetCorrelationValue(m1, m2, 0, 0);
    //R[0][1] = Matrix::GetCorrelationValue(m1, m2, 0, 1);
    //R[1][0] = Matrix::GetCorrelationValue(m1, m2, 1, 0);
    //R[1][1] = Matrix::GetCorrelationValue(m1, m2, 1, 1);


    cout<<R<<endl;

    return 0;
}
