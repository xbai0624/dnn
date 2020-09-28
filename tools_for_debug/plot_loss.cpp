#include <TGraph.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <TApplication.h>
#include <TMultiGraph.h>
#include <TLegend.h>
#include <TAxis.h>

using namespace std;

vector<float> parseLine(string line)
{
    vector<float> res;
    istringstream iss(line);
    float v;
    while(iss>>v)
    {
        res.push_back(v);
    }
    return res;
}

vector<float> parseFile(const char* path)
{
    cout<<"reading file: "<<path<<endl;
    fstream ff(path, fstream::in);
    string line;
    vector<float> res;

    while(getline(ff, line))
    {
        vector<float> tmp = parseLine(line);
	res.insert(res.end(), tmp.begin(), tmp.end());
    }
    return res;
}

TGraph * plot_vector(vector<float> &v, int color)
{
    int N = v.size();
    float *x = new float[N];
    float *y = new float[N];

    for(int i=0;i<N;i++)
    {
        x[i] = i;
	y[i] = v[i];
    }

    TGraph *g = new TGraph(N, x, y);
    g->SetMarkerStyle(20);
    g->SetMarkerColor(color);
    g->SetMarkerSize(1);
    return g;
}

int main(int argc, char* argv[])
{
    TApplication app("app", NULL, NULL);
    if(argc < 1) cout<<"please indicate which file you want to plot."<<endl;

    vector<string> path;
    for(int i=1;i<argc;i++)
    {
	path.push_back(string(argv[i]));
    }

    vector<vector<float>> values;
    for(auto &i: path)
    {
	vector<float> value = parseFile(i.c_str());
	values.push_back(value);
    }

    vector<TGraph*> gs;
    int color = 2;
    for(auto &i: values)
    {
	TGraph *g = plot_vector(i, color);
	color++;
	gs.push_back(g);
    }
    cout<<gs.size()<<endl;
    cout<<"read finished...."<<endl;

    TCanvas *c = new TCanvas("c", "c", 800, 600);
    gPad->SetFrameLineWidth(2);
    //c->SetLogy();

    TMultiGraph *mg = new TMultiGraph();
    for(auto &i: gs)
	mg->Add(i);
    mg->Draw("apl");

    mg->GetXaxis()->SetTitle("Number of Epoch");
    mg->GetXaxis()->CenterTitle();
    mg->GetYaxis()->SetTitle("Loss");
    mg->GetYaxis()->CenterTitle();

    TLegend *leg = new TLegend(0.5, 0.5, 0.7, 0.7);
    for(size_t i=0;i<gs.size();i++)
    {
        leg->AddEntry(gs[i], (path[i].substr(0, path[i].size()-4)).c_str(), "ep");
    }
    leg->Draw();

    app.Run();
    return 0;
}
