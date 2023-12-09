import collections
import time

import torch
import torch.nn as nn


def train_step(model, dataloader, loss_fn, metric, optimizer, scheduler, device):
    model.train()
    dataloader.train()
    metric.reset()

    for X, y in dataloader:
        X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
        loss = loss_fn(model(X).squeeze(), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # type: ignore
        optimizer.step()
        scheduler.step()


def valid_step(model, dataloader, loss_fn, metric, device):
    model.eval()
    dataloader.eval()
    metric.reset()
    loss_val = 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            y_logits = model(X).squeeze()
            loss = loss_fn(y_logits, y)
            loss_val += loss.item()
            metric.update(y_logits.round(), y)

    return loss_val / len(dataloader), metric.compute().item()


def train(
    model, train_loader, valid_loader, loss_fn, metric, optimizer, scheduler, epochs, device
):
    results = collections.defaultdict(list)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_step(model, train_loader, loss_fn, metric, optimizer, scheduler, device)
        t1 = time.perf_counter()

        train_loss, train_acc = valid_step(model, train_loader, loss_fn, metric, device)
        valid_loss, valid_acc = valid_step(model, valid_loader, loss_fn, metric, device)

        print(
            f"Epoch: {epoch:2d} | "
            f"time: {t1 - t0:.2f}s | "
            f"loss: {train_loss:.5f} | "
            f"acc: {train_acc:.5f} | "
            f"val_loss: {valid_loss:.5f} | "
            f"val_acc: {valid_acc:.5f}"
        )

        results["loss"].append(train_loss)
        results["acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)

    return results
