import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mmaction.models.backbones import STGCN
from mmaction.models.heads import GCNHead

POSE_LM, FACE_LM, HAND_LM = 33, 60, 21
V, C = POSE_LM + FACE_LM + 2 * HAND_LM, 4
POSE_OFF, FACE_OFF = 0, POSE_LM
LH_OFF, RH_OFF = FACE_OFF + FACE_LM, FACE_OFF + FACE_LM + HAND_LM
FEATURE_DIM = 438

POSE_EDGES = [
    (0, 11), (0, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (5, 17),
]

def to_nodes(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    T = x.shape[0]
    nodes = np.zeros((T, V, C), dtype=np.float32)
    pose = x[:, :132].reshape(T, 33, 4)
    nodes[:, POSE_OFF:POSE_OFF + 33] = pose

    face = x[:, 132:312].reshape(T, 60, 3)
    nodes[:, FACE_OFF:FACE_OFF + 60, :3] = face
    nodes[:, FACE_OFF:FACE_OFF + 60, 3] = (np.abs(face).sum(axis=-1) > 1e-6).astype(np.float32)

    lh = x[:, 312:375].reshape(T, 21, 3)
    rh = x[:, 375:438].reshape(T, 21, 3)
    nodes[:, LH_OFF:LH_OFF + 21, :3] = lh
    nodes[:, LH_OFF:LH_OFF + 21, 3] = (np.abs(lh).sum(axis=-1) > 1e-6).astype(np.float32)
    nodes[:, RH_OFF:RH_OFF + 21, :3] = rh
    nodes[:, RH_OFF:RH_OFF + 21, 3] = (np.abs(rh).sum(axis=-1) > 1e-6).astype(np.float32)
    return nodes

def sample_sequence(x: np.ndarray, frames: int, training: bool,
                    train_sampling: str, eval_sampling: str,
                    sb_alpha: float, sb_beta: float) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected (T,{FEATURE_DIM}) got {tuple(x.shape)}")
    T = int(x.shape[0])
    if T == 0: return np.zeros((frames, FEATURE_DIM), dtype=np.float32)
    if T == frames: return x
    if T < frames:
        return np.concatenate([x, np.repeat(x[-1:, :], frames - T, axis=0)], axis=0)

    max_start = T - frames
    if training:
        m = train_sampling
        if m == "start_biased":
            r = float(np.random.beta(sb_alpha, sb_beta))
            start = min(int(r * (max_start + 1)), max_start)
        elif m == "random": start = random.randint(0, max_start)
        elif m == "first": start = 0
        elif m == "center": start = max_start // 2
        else: raise ValueError(f"Unknown --train-sampling: {m}")
        return x[start:start + frames]

    m = eval_sampling
    if m == "first": return x[:frames]
    if m == "center":
        start = max_start // 2
        return x[start:start + frames]
    if m == "uniform":
        idx = np.linspace(0, T - 1, frames).round().astype(np.int64)
        return x[idx]
    raise ValueError(f"Unknown --eval-sampling: {m}")

class NpyDataset(Dataset):
    def __init__(self, root: Path, label_map: dict, frames: int, training: bool,
                 train_sampling: str, eval_sampling: str, sb_alpha: float, sb_beta: float):
        self.files = [(p, idx)
                      for lbl, idx in label_map.items()
                      for p in sorted((root / lbl).glob("*.npy"))]
        self.frames, self.training = int(frames), bool(training)
        self.train_sampling, self.eval_sampling = str(train_sampling), str(eval_sampling)
        self.sb_alpha, self.sb_beta = float(sb_alpha), float(sb_beta)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        path, y = self.files[i]
        x = np.load(path).astype(np.float32, copy=False)
        x = sample_sequence(x, self.frames, self.training,
                            self.train_sampling, self.eval_sampling,
                            self.sb_alpha, self.sb_beta)
        x = torch.from_numpy(to_nodes(x)).unsqueeze(0)
        return x, int(y)

def graph_cfg():
    inward = list(POSE_EDGES)
    for u, v in HAND_EDGES:
        inward += [(LH_OFF + u, LH_OFF + v), (RH_OFF + u, RH_OFF + v)]
    return dict(layout=dict(num_node=V, inward=inward, center=0), mode="stgcn_spatial")

class STGCNClassifier(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 64):
        super().__init__()
        bc = int(base_channels)
        self.backbone = STGCN(
            graph_cfg=graph_cfg(), in_channels=C, base_channels=bc,
            num_stages=10, inflate_stages=(5, 8), down_stages=(5, 8), num_person=1
        )
        self.head = GCNHead(num_classes, in_channels=bc * 4)

    def forward(self, x): return self.head(self.backbone(x))

@torch.no_grad()
def topk_correct(out: torch.Tensor, y: torch.Tensor, ks=(1, 5)):
    maxk = min(int(max(ks)), out.size(1))
    _, pred = out.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))
    return [correct[:min(int(k), maxk)].reshape(-1).float().sum().item() for k in ks]

def run_epoch(loader, model, opt=None, amp=False, scaler=None, accum_steps=1, label_smoothing=0.0):
    train = opt is not None
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing) if train else 0.0)
    accum_steps = max(int(accum_steps), 1)
    if train:
        opt.zero_grad(set_to_none=True)
        if amp and scaler is None:
            scaler = torch.amp.GradScaler("cuda", enabled=True)

    tot = cor1 = cor5 = loss_sum = 0.0
    steps = 0
    for x, y in loader:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=bool(amp)):
            out = model(x)
            loss = ce(out, y)
            loss_back = loss / accum_steps

        if train:
            (scaler.scale(loss_back) if amp else loss_back).backward()
            steps += 1
            if steps % accum_steps == 0:
                (scaler.step(opt), scaler.update()) if amp else opt.step()
                opt.zero_grad(set_to_none=True)

        bsz = float(y.size(0))
        loss_sum += float(loss.item()) * bsz
        k1, k5 = topk_correct(out, y, ks=(1, 5))
        cor1 += k1; cor5 += k5; tot += bsz

    if train and (steps % accum_steps) != 0:  # flush remainder
        (scaler.step(opt), scaler.update()) if amp else opt.step()
        opt.zero_grad(set_to_none=True)

    tot = max(tot, 1.0)
    return loss_sum / tot, cor1 / tot, cor5 / tot

def seed_worker(_):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s); random.seed(s)

def make_loader(ds, shuffle, args, g):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=g,
    )

def parse_args():
    ap = argparse.ArgumentParser(description="ST-GCN (MMAction2) with configurable frame sampling.")
    add = ap.add_argument
    add("--data-root", required=True)
    add("--frames", type=int, default=64)
    add("--batch-size", type=int, default=16)
    add("--epochs", type=int, default=50)
    add("--lr", type=float, default=1e-3)
    add("--seed", type=int, default=123)
    add("--num-workers", type=int, default=0)
    add("--train-sampling", type=str, default="start_biased",
        choices=["start_biased", "random", "first", "center"])
    add("--eval-sampling", type=str, default="first",
        choices=["first", "center", "uniform"])
    add("--sb-alpha", type=float, default=1.3)
    add("--sb-beta", type=float, default=4.0)
    add("--amp", action="store_true")
    add("--accum-steps", type=int, default=1)
    add("--base-channels", type=int, default=64)
    add("--weight-decay", type=float, default=1e-2)
    add("--label-smoothing", type=float, default=0.05)
    add("--lr-schedule", type=str, default="cosine", choices=["none", "cosine", "step"])
    add("--step-size", type=int, default=40)
    add("--gamma", type=float, default=0.1)
    add("--best-path", type=str, default="best_val_top1.pt")
    add("--min-samples-per-class", type=int, default=0)
    add("--min-samples-scope", type=str, default="train", choices=["train", "total"])
    add("--early-stop", action="store_true")
    add("--patience", type=int, default=10)
    add("--min-delta", type=float, default=0.0)
    return ap.parse_args()

def count_npys(split_dir: Path, lbl: str) -> int:
    d = split_dir / lbl
    return 0 if not d.exists() else sum(1 for _ in d.glob("*.npy"))

def main():
    a = parse_args()
    torch.backends.cudnn.benchmark = True
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)

    root, train_root = Path(a.data_root), Path(a.data_root) / "train"
    labels_all = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    if not labels_all: raise SystemExit(f"No class folders found under: {train_root}")

    labels = labels_all
    if int(a.min_samples_per_class) > 0:
        min_n = int(a.min_samples_per_class)
        if a.min_samples_scope == "train":
            counts = {l: count_npys(train_root, l) for l in labels_all}
        else:
            sdirs = [root / s for s in ("train", "val", "test") if (root / s).exists()]
            counts = {l: sum(count_npys(sd, l) for sd in sdirs) for l in labels_all}
        labels = [l for l in labels_all if counts.get(l, 0) >= min_n]
        print(f"Class filtering: min={min_n} scope={a.min_samples_scope} kept={len(labels)}/{len(labels_all)} dropped={len(labels_all)-len(labels)}")
        if not labels: raise SystemExit(f"After filtering (min_samples_per_class={min_n}), no classes remain.")

    label_map = {l: i for i, l in enumerate(labels)}
    g = torch.Generator().manual_seed(a.seed)
    mkds = lambda split, train: NpyDataset(root / split, label_map, a.frames, train, a.train_sampling, a.eval_sampling, a.sb_alpha, a.sb_beta)

    train_dl = make_loader(mkds("train", True),  True,  a, g)
    val_dl   = make_loader(mkds("val",   False), False, a, g)

    test_dir = root / "test"
    test_dl = make_loader(mkds("test", False), False, a, g) if test_dir.is_dir() else None
    if test_dl is not None:
        print(f"Found test set: {len(test_dl.dataset)} samples under {test_dir}")

    model = STGCNClassifier(len(labels), base_channels=a.base_channels).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=float(a.weight_decay))

    scheduler = None
    if a.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(a.epochs))
    elif a.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=int(a.step_size), gamma=float(a.gamma))

    scaler = torch.amp.GradScaler("cuda", enabled=bool(a.amp))
    best, best_epoch, no_improve = -1.0, -1, 0
    best_path, patience, min_delta = Path(a.best_path), max(int(a.patience), 1), float(a.min_delta)

    for e in range(1, int(a.epochs) + 1):
        tl, ta1, ta5 = run_epoch(train_dl, model, opt, amp=a.amp, scaler=scaler,
                                 accum_steps=a.accum_steps, label_smoothing=float(a.label_smoothing))
        vl, va1, va5 = run_epoch(val_dl, model, amp=a.amp)

        if va1 > (best + min_delta):
            best, best_epoch, no_improve = float(va1), int(e), 0
            torch.save({"epoch": best_epoch, "val_top1": best,
                        "model": model.state_dict(), "label_map": label_map, "args": vars(a)}, best_path)
        else:
            no_improve += 1

        if scheduler is not None: scheduler.step()
        print(f"Epoch {e:03d}: train loss {tl:.4f} top1 {ta1:.3f} top5 {ta5:.3f} | "
              f"val loss {vl:.4f} top1 {va1:.3f} top5 {va5:.3f} | lr {opt.param_groups[0]['lr']:.2e}")

        if a.early_stop and no_improve >= patience:
            print(f"Early stopping at epoch {e}: no val top1 improvement for {patience} epochs. "
                  f"Best epoch was {best_epoch} (val_top1={best:.3f}).")
            break

    if test_dl is not None and len(test_dl.dataset) > 0:
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            print(f"Loaded best checkpoint from epoch {ckpt.get('epoch')} (val_top1={ckpt.get('val_top1'):.3f}).")
        tloss, t1, t5 = run_epoch(test_dl, model, amp=a.amp)
        print(f"TEST: loss {tloss:.4f} top1 {t1:.3f} top5 {t5:.3f}")

if __name__ == "__main__":
    main()
