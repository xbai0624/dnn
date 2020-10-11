#include <iostream>

#include "Tools.h"
#include "Network.h"

#include "UnitTest.h"

#include <fenv.h>

using namespace std;

int main(int argc, char* argv[])
{
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    Network *net_work = new Network();
    net_work->Init();

    net_work->Train();
    //net_work->Classify();

    // test
    //UnitTest *test = new UnitTest();
    //test->Test();

    cout<<"MAIN TEST SUCCESS!!!"<<endl;
    return 0;
}
