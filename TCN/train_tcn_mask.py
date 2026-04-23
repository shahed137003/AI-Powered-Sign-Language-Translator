import os,gc,json,warnings,pickle,random
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report,confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =========================================================
# CONFIG
# =========================================================
DATA_DIR=r"D:\500 words\data\splits_with_mask"
OUTDIR="tcn_with_mask_results"
os.makedirs(OUTDIR,exist_ok=True)

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE:',DEVICE)

SEED=42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# exact hyperparams from your 91% trial
CHANNELS=128
LAYERS=4
DROPOUT=0.30
LR=1e-3
WD=1e-4
BATCH=32
SMOOTH=0.10
EPOCHS=120
PATIENCE=18

# =========================================================
# GLOSS EXTRACT
# =========================================================
def extract_gloss(name):
    stem=name.replace('.npy','').replace('_mask','').strip()
    parts=stem.rsplit(' ',1)
    if len(parts)==2 and parts[1].isdigit():
        return parts[0].strip()
    return stem

# =========================================================
# LOAD SPLITS
# =========================================================
def load_split(split):

    folder=Path(DATA_DIR)/split
    samples=[]
    masks={}

    for f in folder.glob('*.npy'):
        if '_mask' in f.stem.lower():
            masks[f.stem.replace('_mask','')]=f
        else:
            samples.append(f)

    X=[];M=[];Y=[];F=[]

    for f in sorted(samples):

        arr=np.load(f).astype(np.float32)

        if arr.shape[1]==439:
            x=arr[:,:438]
            m=arr[:,438]
        else:
            x=arr
            if f.stem in masks:
                m=np.load(masks[f.stem]).astype(np.float32)
                if len(m.shape)>1:
                    m=m.squeeze()
            else:
                m=(np.abs(x).sum(-1)>0).astype(np.float32)

        X.append(x)
        M.append(m)
        Y.append(extract_gloss(f.name))
        F.append(f.name)

    return np.stack(X),np.stack(M),np.array(Y),np.array(F)

print('Loading split folders...')
X_train,m_train,l_train,f_train=load_split('train')
X_val,m_val,l_val,f_val=load_split('val')
X_test,m_test,l_test,f_test=load_split('test')

print('Train:',len(X_train))
print('Val:',len(X_val))
print('Test:',len(X_test))

FEATURE_DIM=X_train.shape[2]
TARGET_FRAMES=X_train.shape[1]

# =========================================================
# LABELS
# =========================================================
all_labels=np.concatenate([l_train,l_val,l_test])

le=LabelEncoder()
le.fit(all_labels)

y_train=le.transform(l_train)
y_val=le.transform(l_val)
y_test=le.transform(l_test)

NUM_CLASSES=len(le.classes_)
print('Detected glosses:',NUM_CLASSES)

np.save(os.path.join(OUTDIR,'label_encoder.npy'),le.classes_)

# =========================================================
# DATASET
# =========================================================
class ASLDataset(Dataset):

    def __init__(self,X,M,Y,F,augment=False):
        self.X=torch.tensor(X).float()
        self.M=torch.tensor(M).float()
        self.Y=torch.tensor(Y).long()
        self.F=F
        self.augment=augment

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,i):

        x=self.X[i]
        m=self.M[i]
        y=self.Y[i]
        f=self.F[i]

        if self.augment:
            if torch.rand(1)>.5:
                x=x+torch.randn_like(x)*0.008
            if torch.rand(1)>.5:
                x=x*(0.94+torch.rand(1)*0.12)

        x=x.transpose(0,1)

        return x,m,y,f

train_loader=DataLoader(
    ASLDataset(X_train,m_train,y_train,f_train,True),
    batch_size=BATCH,
    shuffle=True
)

val_loader=DataLoader(
    ASLDataset(X_val,m_val,y_val,f_val),
    batch_size=BATCH
)

test_loader=DataLoader(
    ASLDataset(X_test,m_test,y_test,f_test),
    batch_size=BATCH
)

# =========================================================
# MODEL
# =========================================================
class TemporalBlock(nn.Module):

    def __init__(self,inp,out,dil,drop):
        super().__init__()

        p=dil

        self.net=nn.Sequential(
            nn.Conv1d(inp,out,3,padding=p,dilation=dil),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Conv1d(out,out,3,padding=p,dilation=dil),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.res=nn.Identity() if inp==out else nn.Conv1d(inp,out,1)

    def forward(self,x):
        y=self.net(x)
        if y.size(-1)!=x.size(-1):
            y=y[...,:x.size(-1)]
        return y+self.res(x)

class TCN(nn.Module):

    def __init__(self,input_dim,num_classes,ch,layers,drop):
        super().__init__()

        blocks=[]
        for i in range(layers):
            ind=input_dim if i==0 else ch
            blocks.append(TemporalBlock(ind,ch,2**i,drop))

        self.tcn=nn.Sequential(*blocks)
        self.drop=nn.Dropout(drop)
        self.fc=nn.Linear(ch,num_classes)

    def masked_pool(self,x,m):
        m=m.unsqueeze(1)
        x=x*m
        s=x.sum(-1)
        d=m.sum(-1).clamp(min=1)
        return s/d

    def forward(self,x,m):
        x=self.tcn(x)
        x=self.masked_pool(x,m)
        x=self.drop(x)
        return self.fc(x)

model=TCN(
    FEATURE_DIM,
    NUM_CLASSES,
    CHANNELS,
    LAYERS,
    DROPOUT
).to(DEVICE)

# =========================================================
# LOSS
# =========================================================
weights=compute_class_weight(
'balanced',
classes=np.unique(y_train),
y=y_train
)

weights=torch.tensor(weights).float().to(DEVICE)

class SmoothCE(nn.Module):

    def __init__(self,s=.1):
        super().__init__()
        self.s=s

    def forward(self,logits,target):

        nc=logits.size(1)

        with torch.no_grad():
            td=torch.zeros_like(logits)
            td.fill_(self.s/(nc-1))
            td.scatter_(1,target.unsqueeze(1),1-self.s)

        lp=torch.log_softmax(logits,1)

        w=weights[target]

        loss=-(td*lp).sum(1)*w

        return loss.mean()

criterion=SmoothCE(SMOOTH)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD
)

scheduler=ReduceLROnPlateau(
    optimizer,
    'min',
    factor=.5,
    patience=4
)

# =========================================================
# RUN EPOCH
# =========================================================
def run_epoch(loader,opt=None):

    train=(opt is not None)

    model.train() if train else model.eval()

    tl=0
    correct=0
    total=0

    preds=[]
    truths=[]
    bad=[]

    for x,m,y,f in loader:

        x=x.to(DEVICE)
        m=m.to(DEVICE)
        y=y.to(DEVICE)

        if train:
            opt.zero_grad()

        with torch.set_grad_enabled(train):

            out=model(x,m)
            loss=criterion(out,y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                opt.step()

        tl+=loss.item()*y.size(0)

        p=out.argmax(1)

        correct+=(p==y).sum().item()
        total+=y.size(0)

        preds.extend(p.cpu().numpy())
        truths.extend(y.cpu().numpy())

        if not train:
            for fn,pp,tt in zip(f,p.cpu().numpy(),y.cpu().numpy()):
                if pp!=tt:
                    bad.append([
                        fn,
                        le.inverse_transform([pp])[0],
                        le.inverse_transform([tt])[0]
                    ])

    return tl/total,correct/total,preds,truths,bad

# =========================================================
# NORMAL TRAINING LOOP
# =========================================================
print('Starting training...')

best_val_acc=0
pat_counter=0

tracc=[]
vaacc=[]
trloss=[]
valloss=[]

for ep in range(EPOCHS):

    train_loss,train_acc,_,_,_=run_epoch(train_loader,optimizer)
    val_loss,val_acc,_,_,_=run_epoch(val_loader,None)

    scheduler.step(val_loss)
    lr_now=optimizer.param_groups[0]['lr']

    tracc.append(train_acc)
    vaacc.append(val_acc)
    trloss.append(train_loss)
    valloss.append(val_loss)

    print(
        f'Epoch {ep+1:03d}/{EPOCHS} | '
        f'LR {lr_now:.2e} | '
        f'Train Loss {train_loss:.4f} | '
        f'Train Acc {train_acc:.4f} | '
        f'Val Loss {val_loss:.4f} | '
        f'Val Acc {val_acc:.4f}'
    )

    if val_acc>best_val_acc:

        best_val_acc=val_acc
        pat_counter=0

        torch.save(
            model.state_dict(),
            os.path.join(OUTDIR,'best_model.pth')
        )

        print('Best model saved')

    else:
        pat_counter+=1

    if pat_counter>=PATIENCE:
        print('Early stopping triggered')
        break

# =========================================================
# TEST
# =========================================================
model.load_state_dict(
    torch.load(os.path.join(OUTDIR,'best_model.pth'))
)

tl,ta,preds,truths,bad=run_epoch(test_loader,None)

print('FINAL TEST ACC:',ta)
print(f'TEST ACCURACY %: {ta*100:.2f}')

pd.DataFrame(
    bad,
    columns=['file','prediction','truth']
).to_csv(
    os.path.join(OUTDIR,'wrong_predictions.csv'),
    index=False
)

cm=confusion_matrix(truths,preds)
np.save(os.path.join(OUTDIR,'confusion.npy'),cm)

rep=classification_report(
    truths,
    preds,
    target_names=le.classes_,
    zero_division=0
)

with open(os.path.join(OUTDIR,'report.txt'),'w',encoding='utf8') as f:
    f.write(rep)

with open(os.path.join(OUTDIR,'metrics.json'),'w') as f:
    json.dump(
        {
            'test_acc':float(ta),
            'test_loss':float(tl)
        },
        f,
        indent=2
    )

plt.figure(figsize=(12,5))
plt.plot(tracc,label='train')
plt.plot(vaacc,label='val')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'acc_curve.png'))
plt.close()

print('DONE')
print('Results in:',OUTDIR)
