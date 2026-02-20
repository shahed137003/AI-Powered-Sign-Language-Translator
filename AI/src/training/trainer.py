import torch

def run_epoch(model, loader, criterion, optimizer=None, device="cpu", grad_clip=1.0):
    train = optimizer is not None
    model.train() if train else model.eval()

    loss_sum, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for x, m, y in loader:
            x, m, y = x.to(device), m.to(device), y.to(device)

            out = model(x, m)
            loss = criterion(out, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total