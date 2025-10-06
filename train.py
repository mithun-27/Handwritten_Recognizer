# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_utils import load_datasets
from model import SmallCNN
import json
import argparse
from torchvision import transforms
import torch.nn.functional as F

def train_loop(model, device, loader, optimizer, criterion, epoch):
    model.train()
    running = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        running += loss.item()
    avg = running / len(loader)
    print(f"Train Epoch {epoch}: avg loss {avg:.4f}")

def test_loop(model, device, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            running += criterion(out, target).item()
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = correct / total
    print(f"Test loss {running/len(loader):.4f}, acc {acc:.4f}")
    return acc

def main(args):
    print("Loading datasets (may take a while on first run)...")
    train_ds, test_ds, maps = load_datasets(download=True)
    idx_to_char, char_to_idx = maps

    # Optional: lightweight augmentation on the fly (rotation/shift)
    # We'll wrap datasets with transforms via DataLoader collate if desired.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_classes=36).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loop(model, device, train_loader, optimizer, criterion, epoch)
        acc = test_loop(model, device, test_loader, criterion)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            with open("class_map.json", "w") as f:
                json.dump(idx_to_char, f)
            print(f"Saved best model to {args.save_path} (acc {acc:.4f})")
    print("Training finished. Best acc:", best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-path", type=str, default="model.pt")
    args = parser.parse_args()
    main(args)
