#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

vector<float> parseLine(string _line)
{
    vector<float> res;
    float s;
    istringstream line(_line);
    cout<<"_line: "<<_line<<endl;
    while(line>>s)
    {
        res.push_back(s);
    }
    return res;
}

void get(float &mu, float &var, vector<float> &vec)
{
    for(auto &i: vec) cout<<i<<endl;
    cout<<"...."<<endl;
    mu = 0; var = 0;
    for(auto &i: vec)
        mu += i;
    float s = vec.size();
    cout<<"mu: "<<mu<<", size: "<<s<<endl;
    mu /= s;

    for(auto &i: vec)
    {
        var += (i-mu)*(i-mu);
    }

    var /= s;
    if(s > 1) var = s/(s-1) * var;

    var = sqrt(var);
}

int main(int argc[[maybe_unused]], char* argv[][[maybe_unused]])
{
    string line = "1 0 0";
    vector<float> r = parseLine(line);
    float mu, sigma;
    get(mu, sigma, r);
    cout<<"mu: "<<mu<<", sigma: "<<sigma<<endl;

    cout<<"after: affine transform: "<<endl;
    for(auto &i: r)
        cout<< (i-mu)/sigma<<endl;
    
    return 0;
}
