"""Train and evaluate Hydra. Usage: python train.py --data PATH --epochs 3
Expected: data/train/{real,fake} and data/valid/{real,fake} (or val/ or test/)."""
import argparse, io, json, os, random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from model import HydraNet, IMG, MEAN, STD

device = "cuda" if torch.cuda.is_available() else "cpu"

class RandJPEG:
    def __init__(self, p=0.3): self.p = p
    def __call__(self, im):
        if random.random() > self.p: return im
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(50, 95)); b.seek(0)
        return Image.open(b).convert("RGB")

def jpeg(q):
    def f(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    return f

def downup(s):
    return lambda im: im.resize((int(im.width*s), int(im.height*s))).resize(im.size)

def crop(frac):
    def f(im):
        w,h=im.size; cw,ch=int(w*frac),int(h*frac); l,t=(w-cw)//2,(h-ch)//2
        return im.crop((l,t,l+cw,t+ch))
    return f

def tfm(pre=None, train=False):
    ops=[T.Lambda(lambda im: im.convert("RGB"))]
    if pre: ops.append(T.Lambda(pre))
    if train: ops += [RandJPEG(),T.RandomHorizontalFlip()]
    ops += [T.Resize((IMG,IMG)),T.ToTensor(),T.Normalize(MEAN,STD)]
    return T.Compose(ops)

def make_ds(root,fake_idx,transform):
    return datasets.ImageFolder(root,transform=transform,target_transform=lambda t: float(t==fake_idx))

def pick(n_total,n,seed):
    idx=list(range(n_total)); random.Random(seed).shuffle(idx); return idx[:n]

@torch.no_grad()
def evaluate(model,ds,idx,bs=64):
    model.eval(); ys,ps=[],[]
    for x,y in DataLoader(Subset(ds,idx),batch_size=bs,num_workers=0):
        ps += torch.sigmoid(model(x.to(device))).cpu().tolist(); ys += y.tolist()
    ys,ps=np.array(ys),np.array(ps); pred=(ps>=0.5).astype(int)
    return {"accuracy":accuracy_score(ys,pred),"precision":precision_score(ys,pred,zero_division=0),"recall":recall_score(ys,pred,zero_division=0),"f1":f1_score(ys,pred,zero_division=0),"auc":roc_auc_score(ys,ps),"confusion_matrix":confusion_matrix(ys,pred).tolist()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True); p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--n_train",type=int,default=8000); p.add_argument("--n_val",type=int,default=2000)
    p.add_argument("--bs",type=int,default=32); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    vdir=next(d for d in ("valid","val","test") if os.path.isdir(os.path.join(a.data,d)))
    probe=datasets.ImageFolder(os.path.join(a.data,"train"))
    fake_idx=next(i for c,i in probe.class_to_idx.items() if "fake" in c.lower())
    print("classes:",probe.class_to_idx,"fake index",fake_idx,"device",device)
    tr=make_ds(os.path.join(a.data,"train"),fake_idx,tfm(train=True)); tr_idx=pick(len(tr),a.n_train,a.seed)
    vroot=os.path.join(a.data,vdir); va=make_ds(vroot,fake_idx,tfm()); va_idx=pick(len(va),a.n_val,a.seed)
    loader=DataLoader(Subset(tr,tr_idx),batch_size=a.bs,shuffle=True,num_workers=0,drop_last=True)
    model=HydraNet(pretrained=True).to(device)
    l4=list(model.spatial.layer4.parameters()); rest=list(model.freq.parameters())+list(model.fusion.parameters())
    opt=torch.optim.AdamW([{"params":l4,"lr":1e-4},{"params":rest,"lr":1e-3}],weight_decay=1e-4)
    lossf=nn.BCEWithLogitsLoss(); scaler=torch.amp.GradScaler(enabled=device=="cuda")
    for ep in range(a.epochs):
        model.train(); total=0
        for i,(x,y) in enumerate(loader):
            x,y=x.to(device),y.to(device).float()
            with torch.autocast(device_type=device,enabled=device=="cuda"): loss=lossf(model(x),y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()
            if i%50==0: print(f"epoch {ep+1} step {i}/{len(loader)} loss {loss.item():.4f}",flush=True)
        m=evaluate(model,va,va_idx); print(f"epoch {ep+1}: loss {total/len(loader):.4f} val acc {m['accuracy']:.4f} auc {m['auc']:.4f}")
    results={"clean":evaluate(model,va,va_idx)}
    for name,pre in {"jpeg_q75":jpeg(75),"jpeg_q50":jpeg(50),"resize_0.5x":downup(0.5),"center_crop_80%":crop(0.8)}.items():
        results[name]=evaluate(model,make_ds(vroot,fake_idx,tfm(pre)),va_idx)
    state={k:(v.half() if v.is_floating_point() else v) for k,v in model.state_dict().items()}
    torch.save(state,"hydra.pt"); json.dump(results,open("results.json","w"),indent=2)
    for name,m in results.items(): print(name,m)

if __name__ == "__main__": main()
