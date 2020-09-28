#include <iostream>
#include <random>
#include <atomic>
#include <thread>

using namespace std;

class A
{
public:
    A()
    {
        seed = seed + 1;
    }

    ~A()
    {
    }

    void show()
    {
        int s = seed;
        cout<<s<<endl;
    }
private:
    static atomic<int> seed;
};

atomic<int> A::seed{0};

int main(int argc, char* argv[])
{
    random_device rd;

    A a1[10];

    thread vth[10];
    for(int i=0;i<10;i++)
    {
        vth[i] = thread([&]()->void {a1[i] = A();});
    }

    for(int i=0;i<10;i++)
        vth[i].join();

    for(int i=0;i<10;i++) a1[i].show();


    mt19937 gen(1);

    for(int i=0;i<10;i++)
    cout<<gen()<<endl;

    unsigned int tmp = 491856570;
    cout<<"tmp: = "<<tmp<<endl;

    return 0;
}
