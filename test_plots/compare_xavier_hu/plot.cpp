{
    TRandom *g = new TRandom(0);

    float N = 100.;

    TH1F *hxavier = new TH1F("h", "h", 1000, -1, 1);
    // xavier
    for(int i=0;i<1e6;i++)
    {
        hxavier->Fill(g->Gaus(0, 1./TMath::Sqrt(N)));
    }

    ///hxavier->Draw();

    // kaiming hu
    TH1F *hhu = new TH1F("hu", "hu", 1000, -1, 1);
    for(int i=0;i<1e6;i++)
    {
        hhu->Fill( g->Gaus(0, 1) * TMath::Sqrt(2./N));
    }

    TCanvas *c = new TCanvas("c", "c", 800, 600);
    hxavier->Draw();
    hhu->SetLineColor(2);
    hhu->Draw("same");
}
