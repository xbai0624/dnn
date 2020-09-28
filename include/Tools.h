#ifndef TOOLS_H
#define TOOLS_H

// here declares some commonly used routines

#include <vector>
#include <iostream>

#include <algorithm> // std::shuffle
#include <random> //std::default_random_engine
#include <chrono> // std::chrono::sytem_clock

namespace TOOLS
{
    template<class T>
	void Shuffle(std::vector<T> &vec) 
	{
	    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	    shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));
	}

    template<class T1, class T2>
	void Shuffle(std::vector<T1> &vec1, std::vector<T2> &vec2) 
	{
	    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	    shuffle(vec1.begin(), vec1.end(), std::default_random_engine(seed));
	    shuffle(vec2.begin(), vec2.end(), std::default_random_engine(seed));
	}

    template<typename T>
	void INFO(T info)
	{
	    std::cout<<" #INFO#: "<<info<<std::endl;
	}

    template<typename T, typename... Args>
	void INFO(T info, Args... others)
	{
	    std::cout<<" #INFO#: "<<info<<", ";
	    return INFO(others...);
	}
};

#endif
