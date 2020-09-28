#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>

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

vector<float> parseFile(const char* path)
{
    fstream ff(path, fstream::in);
    string line;
    vector<float> res;

    while(getline(ff, line))
    {
        vector<float> tmp = parseLine(line);
	res.insert(res.end(), tmp.begin(), tmp.end());
    }
    return res;
}

float mul(vector<float> &v1, vector<float> &v2)
{
    assert(v1.size() == v2.size());

    float tmp = 0.;
    for(size_t i=0;i<v1.size();i++)
    {
        tmp += v1[i]*v2[i];
    }
    return tmp;
}

int main(int argc, char* argv[])
{
    vector<float> v1 = parseFile("v1.txt");
    vector<float> v2 = parseFile("v2.txt");

    float m = mul(v1, v2);

    cout<<"matrix : 1"<<endl;
    for(auto &i: v1) cout<<i<<", ";
    cout<<endl<<endl<<endl;
    cout<<"matrix : 2"<<endl;
    for(auto &i: v2) cout<<i<<", ";
    cout<<endl<<endl<<endl;
    cout<<"product = : "<<m<<endl;

    return 0;
}
