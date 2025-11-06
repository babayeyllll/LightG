# Extracted portions of the training script provided for review.
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

BATCH = 4096
NUM_WORKERS = 0
USE_AMP = True
loss_fn = nn.SmoothL1Loss(reduction="none")
model = nn.Linear(16, 1)
opt = torch.optim.AdamW(model.parameters())
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)


class DummySet(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        wb = torch.ones(1)
        xb = torch.randn(16)
        yb = torch.randn(1)
        return xb, yb, wb


def collate(batch):
    xb, yb, wb = zip(*batch)
    return (
        torch.stack(xb),
        torch.stack(yb),
        torch.stack(wb),
    )


dl_tr = DataLoader(
    DummySet(),
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate,
    prefetch_factor=(4 if NUM_WORKERS > 0 else None),
)


def training_step(xb, yb, wb):
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        pred = model(xb)
        loss = (loss_fn(pred, yb) * wb).mean()
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return loss


if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("未检测到第 2 张 GPU。需要 cuda:1。")

torch.cuda.set_device(1)
