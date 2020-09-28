#include <iostream>

int main(int argc, char* argv[])
{
    float a;
    std::cout<<"input your a value: "<<std::endl;
    std::cin>>a;
    std::cout<<" sigma^prime = a(1-a) = : "<<a*(1.-a)<<std::endl;
    return 0;
}
