import torch
import torch.nn as nn
import numpy as np
from .config import MODEL_PATH, LABEL_PATH, DEVICE #get the paths
import sys
from pathlib import Path

# add preprocessing path
from .config import PREPROCESSING_PATH
sys.path.append(str(PREPROCESSING_PATH))
from preprocessing.pipeline_v3 import preprocess_sequence_global
from preprocessing.constants import FEATURE_DIM


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

class ModelWrapper:
    def __init__(self):
        self.labels = np.load(LABEL_PATH, allow_pickle=True)
        self.labels = np.array(self.labels).astype(str)

        self.model = TCN(
            FEATURE_DIM,
            len(self.labels),
            ch=128,
            layers=4,
            drop=0.3
        ).to(DEVICE)

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        # safer loading (handles both save formats)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def predict(self, sequence):
        """
        sequence: (T, F)
        """

        # convert input to tensor (F, T)
        x = torch.from_numpy(sequence).float().transpose(0, 1).unsqueeze(0)

        # mask (same logic as pipeline)
        m = (sequence.sum(axis=1) != 0).astype(np.float32)
        m = torch.from_numpy(m).float().unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x, m)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        label = self.labels[idx.item()]
        return label, conf.item() * 100