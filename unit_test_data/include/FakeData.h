/*
 * generate double-gaussian signals and linear cosmics
 */

#ifndef FAKE_DATA_H
#define FAKE_DATA_H

#include <string>
#include <fstream>


class FakeData{
public:
    FakeData();
    ~FakeData();

    void Generate(const char *path, std::string type, size_t N, float energy);
    void GenerateGaussian();
    void GenerateCosmic();

    void SetBinWidth(float w){bin_width = w;}
    float GetBinWidth(){return bin_width;}


private:
    size_t N_events;
    const char *path;

    // hycal settings
    float bin_width = 17.0; // mm hycal block size: 17mm
    size_t signal_nbins = 3; // signal size: 3x3
    size_t hycal_size = 4; // hycal size: 4x4
    float __energy;

    std::fstream __file;
};

#endif
