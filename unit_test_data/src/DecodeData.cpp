#include "DecodeData.h"

#include <iostream>
#include <sstream>

DecodeData::DecodeData(){
}

DecodeData::~DecodeData(){
}

std::vector<std::string> DecodeData::ParseLine(std::string line)
{
    std::vector<std::string> res;
    if(line.size() <= 0) return res;

    std::istringstream iss(line);
    std::string tmp;
    while(std::getline(iss, tmp, ' '))
    {
        if(tmp.size() > 0) res.push_back(tmp);
    }
    return res;
}

void DecodeData::ReadEntry(std::string line){
    auto __c = ParseLine(line);

    if(__c.size() <= 0) return;
    if( __c.size()%3 != 0) {
        std::cout<<"Warning: corrupted data skipped..."<<std::endl;
	std::cout<<"         size: "<<__c.size()<<std::endl;
	std::cout<<"\""<<line<<"\""<<std::endl;
	return;
    }

    std::vector<Entry> points;

    for(size_t i=0; i<__c.size(); i+=3){
        Entry v;
        v.i = std::stoi(__c[i]);
	v.j = std::stoi(__c[i+1]);
	v.val = std::stod(__c[i+2]);
        points.push_back(v);
    }

    __data.push_back(points);
}

void DecodeData::ReadAllEntries(const char *path){
    __file.open(path);
    if(!__file.is_open()) {
        std::cout<<"Error: "<<path<<" cannot open."<<std::endl;
	exit(0);
    }

    std::string __tmp;
    while(std::getline(__file, __tmp)){
        ReadEntry(__tmp);
    }
}

std::vector<Entry> DecodeData::GetEntry(size_t i){
    if(__data.size() <= 0) {
        std::cout<<"Warning: event pool empty."<<std::endl;
        exit(0);
    }
    if(__data.size() <= i ){
        std::cout<<"Warning: exceeded event pool."<<std::endl;
        return __data[0];
    }
    return __data[i];
}
