#include <iostream>
#include <math.h>

using namespace std;

int main(int argc, char* argv[])
{
    float a;
    cout<<"input your W*A value: "<<endl;
    cin>>a;

    float b;
    cout<<"input your bias value: "<<endl;
    cin>>b;

    a = a + b;
    cout<<"Z = : "<<a<<endl;
   
    cout<<"sigmoid(z): = "<< 1./ (1. + exp(-a))<<endl;
    return 0;
}
