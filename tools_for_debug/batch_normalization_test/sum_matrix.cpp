#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char* argv[])
{
    fstream f("data.txt", fstream::in);

    double res = 0;

    string line;
    while(getline(f, line))
    {
        istringstream iss(line);
	string tmp;
	while(iss>>tmp)
	{
	    double d = stod(tmp);
	    res += d;
	}
    }

    cout<<res<<endl;
    cout<<res/10./16.<<endl;

    return 0;
}
