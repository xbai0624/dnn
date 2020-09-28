
vector<string> GetFiles()
{
    vector<string> res;
    fstream f("list.txt", fstream::in);
    string line;
    while(getline(f, line))
    {
        res.push_back(line);
    }
    return res;
}

vector<vector<float>> GetWB(const char* path)
{
    vector<vector<float>> res;
    
    fstream f(path, fstream::in);
    string line;
    vector<float> tmp;
    while(getline(f, line))
    {
        if(line.size() <= 0) continue;

	if(line.find("weight") != string::npos || line.find("bias")!= string::npos)
	{
	    if(tmp.size()>0) res.push_back(tmp);
	    tmp.clear();
	}
	else 
	{
	    istringstream iss(line);
	    float v;
	    while(iss>>v) tmp.push_back(v);
	}
    }
    if(tmp.size() > 0) res.push_back(tmp);

    return res;
}

vector<TH1F*> PlotHistos(vector<vector<float>> wb, string layer)
{
    vector<TH1F*> res;
    for(size_t i=0;i<wb.size();i++)
    {
        TH1F *h = new TH1F(Form("%s_h%d", layer.c_str(), (int)i), Form("%s_h%d", layer.c_str(), (int)i), 500, -2, 2);
	for(auto &v: wb[i])
	    h->Fill(v);
	res.push_back(h);
    }
    return res;
}

void histo()
{
    vector<string> files = GetFiles();

    for(size_t i=0;i<files.size();i++)
    {
	vector<vector<float>> wb = GetWB(files[i].c_str());
	vector<TH1F*> _histos = PlotHistos(wb, files[i].substr(32, 8));

	TCanvas *c = new TCanvas(Form("c%d", (int)i), Form("c%d", (int)i), 1000, 900);
	int n_histos = _histos.size();
	int N = TMath::Sqrt(n_histos) + 1;
	c->Divide(N, N);
	for(size_t k=0;k<_histos.size();k++)
	{
	    c->cd(k+1);
	    _histos[k]->Draw();
	}
    }
}
