#include <iostream>
#include <vector>

#include "FakeData.h"
#include "DecodeData.h"

#include <TFile.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TApplication.h>
#include <TStyle.h>

using namespace std;

int main(int argc, char* argv[]){
    //TApplication theApp("App", &argc, argv);

    FakeData *fd = new FakeData();
    //fd->Generate("data.dat", "gaussian", 100, 500);
    //fd->Generate("data_cosmic.dat", "cosmic", 1000, 1000); // (file_name, event_type, number_of_events, energy)
    //fd->Generate("data_signal.dat", "gaussian", 1000, 1000); // (file_name, event_type, number_of_events, energy)
    fd->Generate("data_signal.dat", "gaussian", 2000, 1000); // (file_name, event_type, number_of_events, energy)


    //DecodeData *dat = new DecodeData();
    //dat->ReadAllEntries("data.dat");

/*
    TCanvas *c = new TCanvas("c", "c", 900, 800);
    gStyle->SetOptStat(0);

    for(int k=0;k<10;k++)
    {
	vector<Entry> sample = dat->GetEntry(k);
	TH2F *hh = new TH2F(Form("hh%d", k), "hh", 10, -5.*17, 5*17, 10, -5.*17, 5.*17);
	for(auto &i: sample){
	    hh->SetBinContent(i.i, i.j, i.val);
	}
	c->cd();
	//hh->Draw("LEGO2Z");
	hh->Draw("colz");
	//c->SetLogz();
	c->Update();
	getchar();
    }
*/
    cout<<"SUCCESS!!!!"<<endl;
    //theApp.Run();
    return 0;
}
