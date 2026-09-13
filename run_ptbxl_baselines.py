import argparse, random, time, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, p, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
    }

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(), nn.Linear(256*96,512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512,5))
    def forward(self,x): return self.net(x)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(96,128,7,padding=3),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),
                               nn.Conv1d(128,256,5,padding=2),nn.BatchNorm1d(256),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Linear(256,5)
    def forward(self,x): return self.fc(self.net(x.transpose(1,2)).squeeze(-1))

class Block(nn.Module):
    def __init__(self,c): super().__init__(); self.b=nn.Sequential(nn.Conv1d(c,c,3,padding=1),nn.BatchNorm1d(c),nn.ReLU(),nn.Conv1d(c,c,3,padding=1),nn.BatchNorm1d(c)); self.r=nn.ReLU()
    def forward(self,x): return self.r(x+self.b(x))

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv1d(96,128,7,padding=3),nn.BatchNorm1d(128),nn.ReLU())
        self.blocks=nn.Sequential(Block(128),Block(128),nn.MaxPool1d(2),Block(128),Block(128),nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Linear(128,5)
    def forward(self,x): return self.fc(self.blocks(self.stem(x.transpose(1,2))).squeeze(-1))

class GRU(nn.Module):
    def __init__(self): super().__init__(); self.gru=nn.GRU(96,128,batch_first=True,bidirectional=True); self.fc=nn.Linear(256,5)
    def forward(self,x): _,h=self.gru(x); return self.fc(torch.cat([h[-2],h[-1]],1))

class Linear(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Flatten(), nn.Linear(256*96,5))
    def forward(self,x): return self.net(x)

def make_model(name):
    return {"linear":Linear, "mlp":MLP, "cnn":CNN1D, "resnet":ResNet1D, "gru":GRU}[name]()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; losses=[]; crit=nn.BCEWithLogitsLoss()
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        out=model(x); loss=crit(out,y)
        losses.append(loss.item()); ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(out).cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); m=metrics(y,p); m["loss"]=float(np.mean(losses)); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", choices=["linear","mlp","cnn","resnet","gru"], required=True)
    ap.add_argument("--data_dir", default="datasets/ecg_datasets/PTBXL_QRS/superdiagnostic")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baseline_results.csv")
    args=ap.parse_args(); seed_all(args.seed); device="cuda" if torch.cuda.is_available() else "cpu"
    d=Path(args.data_dir)
    Xtr=np.load(d/"train_data.npy").astype("float32"); ytr=np.load(d/"train_labels.npy").astype("float32")
    Xv=np.load(d/"val_data.npy").astype("float32"); yv=np.load(d/"val_labels.npy").astype("float32")
    Xt=np.load(d/"test_data.npy").astype("float32"); yt=np.load(d/"test_labels.npy").astype("float32")
    mean=Xtr.mean(); std=Xtr.std()+1e-6
    Xtr=(Xtr-mean)/std; Xv=(Xv-mean)/std; Xt=(Xt-mean)/std
    tr=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=args.batch_size,shuffle=True,num_workers=4)
    va=DataLoader(TensorDataset(torch.tensor(Xv),torch.tensor(yv)),batch_size=args.batch_size,shuffle=False,num_workers=4)
    te=DataLoader(TensorDataset(torch.tensor(Xt),torch.tensor(yt)),batch_size=args.batch_size,shuffle=False,num_workers=4)
    model=make_model(args.model).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.05); crit=nn.BCEWithLogitsLoss()
    best=-1; best_state=None; start=time.time()
    for ep in range(1,args.epochs+1):
        model.train()
        for x,y in tr:
            x=x.to(device); y=y.to(device); opt.zero_grad(); loss=crit(model(x),y); loss.backward(); opt.step()
        val=evaluate(model,va,device)
        print(f"epoch {ep}: val_auc={val['auc']:.6f} val_f1={val['f1']:.6f} val_loss={val['loss']:.6f}")
        if val["auc"]>best: best=val["auc"]; best_state={k:v.cpu() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); test=evaluate(model,te,device); runtime=time.time()-start
    Path("results").mkdir(exist_ok=True)
    write_header=not Path(args.out).exists()
    with open(args.out,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["model","seed","epochs","batch_size","lr","best_val_auc","test_auc","test_accuracy","test_precision","test_recall","test_f1","test_loss","seconds"])
        if write_header: w.writeheader()
        w.writerow({"model":args.model,"seed":args.seed,"epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,"best_val_auc":best,"test_auc":test["auc"],"test_accuracy":test["accuracy"],"test_precision":test["precision"],"test_recall":test["recall"],"test_f1":test["f1"],"test_loss":test["loss"],"seconds":round(runtime,2)})
    print("TEST", test); print("saved to", args.out)

if __name__=="__main__": main()
