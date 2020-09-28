#include <iostream>
#include <math.h>

using namespace std;

int main(int argc, char* argv[])
{
    float a;
    cout<<"input your z1 value: "<<endl;
    cin>>a;

    float b;
    cout<<"input your z2 value: "<<endl;
    cin>>b;

    a = exp(a);
    b = exp(b);

    cout<<"a1 = : "<<a/(a+b)<<endl;
     cout<<"a2 = : "<<b/(a+b)<<endl;
   
    return 0;
}
