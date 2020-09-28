#include <iostream>
#include <TH1F.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <TLegend.h>

using namespace std;

vector<double> ReadFile(const char* path)
{
    vector<double> res;
    fstream f(path, fstream::in);
    string line;
    while(getline(f, line))
    {
        istringstream iss(line);
	string tmp;
	while(iss>>tmp)
	{
	    double a = stod(tmp);
	    res.push_back(a);
	}
    }

    return res;
}

int main(int argc, char* argv[])
{
    TApplication *app = new TApplication("app", &argc, argv);

    vector<double> image_before = ReadFile("image_before.txt");
    vector<double> image_after = ReadFile("image_after.txt");

    auto fill_histo = [&](TH1F* h, const vector<double> &vec){
        for(auto &i: vec) h->Fill(i);
    };

    TH1F *h_before = new TH1F("h_before", "h_before", 200, -10, 10);
    TH1F *h_after = new TH1F("h_after", "h_after", 200, -10, 10);

    fill_histo(h_before, image_before);
    fill_histo(h_after, image_after);

    h_before->SetLineColor(2);
    h_before->SetLineWidth(2);
    h_after->SetLineWidth(2);

    cout<<"before entries: "<<h_before->GetEntries()<<endl;
    cout<<"after entries: "<<h_after->GetEntries()<<endl;

    // average and variance
    cout<<"before average and variance: "<<endl;
    cout<<"average: "<<h_before->GetMean()<<endl;
    cout<<"rms: "<<h_before->GetRMS()<<endl;
    cout<<"after average and variance: "<<endl;
    cout<<"average: "<<h_after->GetMean()<<endl;
    cout<<"rms: "<<h_after->GetRMS()<<endl;


    TCanvas *c = new TCanvas("c", "c", 800, 500);
    gPad->SetFrameLineWidth(2);
    h_before->Draw();
    h_after->Draw("same");

    TLegend *leg = new TLegend(0.55, 0.65, 0.88, 0.75);
    leg->AddEntry(h_before, "before batch normalization", "lep");
    leg->AddEntry(h_after, "after batch normalization", "lep");
    leg->Draw();

    app->Run();
    return 0;
}
