import os
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from models.vit import VisionTransformer
from data.data_loader import get_tinyimagenet_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Early Stopping Utility
# -------------------------

class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False

        else:
            self.counter += 1

            if self.counter >= self.patience:
                return True

            return False


# -------------------------
# Train One Epoch
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion,config):

    model.train()

    running_loss = 0
    correct = 0
    total = 0
    print("current config",config)
    for images, labels in tqdm(loader):
        
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)

    accuracy = 100 * correct / total

    return avg_loss, accuracy


# -------------------------
# Validation
# -------------------------

def validate(model, loader, criterion,config):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        print("current config",config)
        for images, labels in tqdm(loader):
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)

    accuracy = 100 * correct / total

    return avg_loss, accuracy


# -------------------------
# Training Pipeline
# -------------------------

def train_model(
    patch_size,
    emb_dim,
    heads,
    depth,
    mlp_dim,
    epochs=10,
    lr=3e-4,
    dropout=0.1,
    CHECKPOINT_DIR="checkpoints",
    BEST_MODEL_DIR="best_models",
    config="default"
):

    data_dir = "data/tiny-imagenet-200"

    train_loader, val_loader = get_tinyimagenet_dataloaders(
        data_dir=data_dir,
        batch_size=64
    )

    model = VisionTransformer(
        img_size=64,
        patch_size=patch_size,
        num_classes=200,
        emb_dim=emb_dim,
        depth=depth,
        num_heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    # -------------------------
    # Learning Rate Scheduling
    # -------------------------

    warmup_epochs = 2

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    early_stopping = EarlyStopping(patience=5)


    best_val_acc = 0
    best_train_acc = 0
    
    log_file = f"training_log_{config}.csv"

    with open(log_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            config
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        with open(log_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc]
            )

        checkpoint_path = os.path.join(
                                        CHECKPOINT_DIR,
                                        f"epoch_{epoch}_{config}.pt"
                                    )


        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config
            },
            checkpoint_path
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_path = os.path.join(
                                            BEST_MODEL_DIR,
                                            f"best_model_{config}.pt"
                                        )


            torch.save(model.state_dict(), best_model_path)

            print("Best model saved")
        if train_acc > best_train_acc:

            best_train_acc = train_acc
            best_train_loss = train_loss
        if early_stopping(val_loss):

            print("Early stopping triggered")

            break

    return best_val_acc, best_val_loss, best_train_acc, best_train_loss


if __name__ == "__main__":

    train_model(
        patch_size=4,
        emb_dim=128,
        heads=4,
        depth=6,
        mlp_dim=256
    )
