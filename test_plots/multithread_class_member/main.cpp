#include <iostream>
#include <thread>
#include <vector>

using namespace std;

class C
{
public:
    void Set(int i, double v )
    {
        a[i] = v;
    }

    void work()
    {
	vector<thread> vth;

	auto func = [&](int i, double v)
	{
	    Set(i, v);
	};

	for(int i=0;i<4;i++)
	{
	    //vth.push_back( thread(&C::Set, this, i, (i+1)*0.02) );
	    vth.push_back(thread(func, i, (i+1)*0.02));
	}
	for(auto &i: vth)
	    i.join();
    }

    void print()
    {
	for(auto &i: a)
	    cout<<i<<endl;
    }

private:
    double a[4];
};

int main(int argc, char* argv[])
{
    C* c = new C();
    c->work();
    c->print();

    return 0;
}
