import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.configs.tcn_config import *
from src.data.dataloaders import build_dataloaders
from src.models.sign2text.tcn import TCN
from src.models.losses import SmoothCE
from src.training.trainer import run_epoch

def main():

    train_loader, val_loader, test_loader, num_classes = build_dataloaders()

    model = TCN(FEATURE_DIM, num_classes).to(DEVICE)

    criterion = SmoothCE(LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, EPOCHS)

    best, patience = -1e9, 0

    for e in range(EPOCHS):
        tr_l, tr_a = run_epoch(model, train_loader, criterion, optimizer, DEVICE, GRAD_CLIP)
        va_l, va_a = run_epoch(model, val_loader, criterion, None, DEVICE)

        scheduler.step()

        metric = va_a - va_l
        print(f"E{e+1:03d} | TL {tr_l:.3f} TA {tr_a:.3f} | VL {va_l:.3f} VA {va_a:.3f}")

        if metric > best:
            best = metric
            patience = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                break

if __name__ == "__main__":
    main()