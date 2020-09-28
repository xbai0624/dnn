#ifndef DECODE_DATA_H
#define DECODE_DATA_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <fstream>

struct Entry {
    float val;
    size_t i;
    size_t j;

    Entry(){
        i = -999; j=-999; val=-9999;
    }
};

class DecodeData {
public:
    DecodeData();
    ~DecodeData();

    void ReadAllEntries(const char* path);

    std::vector<Entry> GetEntry(size_t);
    std::vector<std::string> ParseLine(std::string);
    void ReadEntry(std::string);


private:
    std::vector<std::vector<Entry>> __data;

    std::fstream __file;
};

#endif
