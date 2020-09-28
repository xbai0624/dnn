#include "FakeData.h"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <utility>
#include <random>

#include <TMath.h>
#include <Math/GSLRndmEngines.h>

static float gaussian2D(float x, float y, float amp, float rho=0.){
    // a 2d gaussian function, for 5x5 signal size, bin_width = 17.0;
    // offset at 0, widths are sigma_x, sigma_y respectively
    // return the signal strength at (x, y), amplitude default to 1.0
    // correlation between x and y is rho, default value is 0
    float r = 1.0*17.0;
    float sigma_x = 1.0;
    float sigma_y = sigma_x;

    float res;
    float z;
    x /=r; y/=r;

    z = x*x/sigma_x/sigma_x + y*y/sigma_y/sigma_y - 2.0*rho*x*y/sigma_x/sigma_y;
    res = 1.0/(2*TMath::Pi()*sigma_x*sigma_y*TMath::Sqrt(1-rho*rho)) * TMath::Exp(-z/2.0/(1-rho*rho) );
    return amp*res;
}

// block position in one signal
// signal size 3x3, origin at 0
static const float __x[3] = {
     -17.0, 0, 17.0
};
static const float __y[3] = {
     -17.0, 0, 17.0
};


FakeData::FakeData(){
    // place holder
}

FakeData::~FakeData(){
    // place holder
}

void FakeData::Generate(const char *_path, std::string type, size_t N, float energy){
    N_events = N;
    path = _path;
    __energy = energy;

    if(type == "gaussian")
	GenerateGaussian();
    else if(type == "cosmic" )
	GenerateCosmic();
    else{	
	std::cout<<"Error: unsupported signal shape."<<std::endl;
	exit(0);
    }
}

void FakeData::GenerateGaussian(){
    ROOT::Math::GSLRandomEngine *g = new ROOT::Math::GSLRandomEngine();
    g->Initialize();

    //--------- signal strength
   // amplitude
    float amp = __energy;
    float __sig[3][3];
    //float saved_amp = 0;
    for(size_t i=0;i<3;i++){
        for(size_t j=0;j<3;j++){
	    float val = gaussian2D(__x[i], __y[j], amp);
	    __sig[i][j] = val;
	    //saved_amp += val;
	}
    }
    //std::cout<<"saved amplitude: "<<saved_amp<<", amp: "<<amp<<std::endl;

    //---------- open event file 
    //__file.open(path, std::fstream::out|std::fstream::app);
    __file.open(path, std::fstream::out);

    //-------- signal position
    // sigma
    float R = hycal_size*bin_width/2.0;
    double sigma_x = 1./8 * R;
    double sigma_y = sigma_x;
    // position
    double x_pos, y_pos;
    // correlation
    double rho = 0; // gaussian2d, rho=0
 
    //---------- generate events
    for(size_t i=0; i<N_events;i++){
        // get the position of the hit
	g->Gaussian2D(sigma_x, sigma_y, rho, x_pos, y_pos); // signal position
	// generate hit and save it to txt file
        int I = (x_pos + hycal_size/2.0 * bin_width)/bin_width;
	int J = (y_pos + hycal_size/2.0 * bin_width)/bin_width;

        I = 0; J = 0; // put it at center

	for(size_t ii=0;ii<3;ii++){
	    for(size_t jj=0;jj<3;jj++){

	        float val = __sig[ii][jj];
		val += g->Gaussian(val/5.); // disturb signal by a gaussian

	        __file<<std::setfill(' ')<<std::setw(4)<<I+ii
		    <<std::setfill(' ')<<std::setw(4)<<J+jj
		    <<std::setfill(' ')<<std::setw(14)<<std::setprecision(6)<<val;
	    }
	}
	__file<<std::endl;
    }

}

static float __linear_x(float x, float x_intercept, float slope){
    // a linear function, x is known, solve y
    //  y = k(x - x0)
    return slope*(x - x_intercept);
}

static float __linear_y(float y, float x_intercept, float slope){
    // a linear function, y is know, solve x
    return y/slope + x_intercept;
}

static std::vector<size_t> __get_bin_range(float x0, float slope, bool vertical, float bin_width, float l){
    std::vector<size_t> res;

    float a0, a1;
    if(vertical) {
	a0 = __linear_x(0, x0, slope);
	a1 = __linear_x(l, x0, slope);
    }
    else {
	a0 = __linear_y(0, x0, slope); 
	a1 = __linear_y(l, x0, slope);
    }

    float _min = a0<a1?a0:a1;
    float _max = a0>a1?a0:a1;
    if(_min > l) return res;
    if(_max < 0) return res;

    _min = 0>_min?0:_min;
    _max = l>_max?_max:l;

    size_t n = (_max - _min)/bin_width;
    size_t n_start = _min/bin_width;
    for(size_t i = 0;i<n;i++)
        res.push_back(i + n_start);
    return res;
}

static float __length(float x0, float slope, float x1, float x2, float y1, float y2){
    // get the length of the path that the line traverse through
    // in an area (x1, y1) -> (x2, y2)
    float _y_0 = __linear_x(x1, x0, slope);
    float _y_1 = __linear_x(x2, x0, slope);
    float _y_min = _y_0<_y_1?_y_0:_y_1;
    if(_y_min >= y2) return 0;
    float _y_max = _y_0>_y_1?_y_0:_y_1;
    if(_y_max <= y1) return 0;

    _y_min = _y_min>y1?_y_min:y1;
    _y_max = _y_max<y2?_y_max:y2;

    float _x_0 = __linear_y(y1, x0, slope);
    float _x_1 = __linear_y(y2, x0, slope);
    float _x_min = _x_0<_x_1?_x_0:_x_1;
    if(_x_min >= x2) return 0;
    float _x_max = _x_0>_x_1?_x_0:_x_1;
    if(_x_max <= x1) return 0;

    _x_min = _x_min>x1?_x_min:x1;
    _x_max = _x_max<x2?_x_max:x2;

    float res = TMath::Sqrt((_x_max-_x_min)*(_x_max-_x_min) + (_y_max-_y_min)*(_y_max-_y_min));
    return res;
}

struct pair_hash {
    //to use pair as key of unordered_map, b/c default unordered_map cannot hash pair
    template<class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &x) const {
        return std::hash<T1>()(x.first) ^ std::hash<T2>()(x.second);
    }
};

static std::unordered_map<std::pair<size_t, size_t>, float, pair_hash> __fired_modules(float x0, float slope, float bin_width, float hycal_length, float energy){
    std::unordered_map<std::pair<size_t, size_t>, float, pair_hash> res;

    auto x_range = __get_bin_range(x0, slope, false, bin_width, hycal_length);
    auto y_range = __get_bin_range(x0, slope, true, bin_width, hycal_length);

    if(x_range.size() == 0 || y_range.size() == 0) return res;
    float unit_energy = energy/34.0/1.4; // energy in each module is proportional to its length in that module

    for(auto &i: x_range){
        float x1 = i*bin_width;
	float x2 = (i+1)*bin_width;
        for(auto &j: y_range){
	    float y1 = j*bin_width;
	    float y2 = (j+1)*bin_width;
	    float length = __length(x0, slope, x1, x2, y1, y2);
	    float val = length/bin_width*unit_energy;
	    if(val > 0)res.emplace(std::pair<size_t, size_t>(i, j), val);
	}
    }
    return res;
}

void FakeData::GenerateCosmic()
{
    // cosmic theta angle range (-30deg, 30deg), relative to vertical axis
    // cosmic will introduce ADC signals in the hycal modules it passes though,
    // the ADC strength are proportional to the length of its path in that module
    //
    // so all we need to do is to find the module index on the path, and the length
    std::random_device rd;
    std::mt19937 mt(rd());

    //---------- open event file 
    //__file.open(path, std::fstream::out|std::fstream::app);
    __file.open(path, std::fstream::out);

    // (x0, 0);
    float l = hycal_size * bin_width;
    float x_intercept_low = - l/1.732;
    float x_intercept_high = l*(1 + 1/1.732);
    std::uniform_real_distribution<float> dist_x0(x_intercept_low, x_intercept_high);

    // slope
    float angle_low = 1./3.*TMath::Pi();
    float angle_high = 2./3.*TMath::Pi();
    std::uniform_real_distribution<float> dist_slope(angle_low, angle_high);

    // disturb adc by a normal distribution
    std::normal_distribution<float> normal_dist(0, 1.0);

    std::cout<<"generating: "<<N_events<<" cosimc signals..."<<std::endl;
    for(size_t i=0;i<N_events;i++)
    {
	float x0 = dist_x0(mt);
	float slope = TMath::Tan(dist_slope(mt));

	auto data = __fired_modules(x0, slope, bin_width, bin_width*hycal_size, __energy);
	if(data.size() != 5){ i--; continue; } // only keep events with  9 modules fired
	for(auto &i: data){
	    float adc = i.second + normal_dist(mt);
	    if(adc < 0 ) adc = 0;
	    __file<<std::setfill(' ')<<std::setw(4)<<i.first.first
		<<std::setfill(' ')<<std::setw(4)<<i.first.second
		<<std::setfill(' ')<<std::setw(14)<<std::setprecision(6)<<adc;
	}
	__file<<std::endl;
    }
}

