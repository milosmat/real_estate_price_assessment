import os
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_files(image_root: str, folders: Optional[List[str]] = None, recursive: bool = True) -> List[str]:
    """Return list of image file paths.

    - If folders provided, will search only those subfolders under image_root (non-recursive).
    - If folders is None and recursive=True, will walk all subdirectories under image_root.
    - Supports jpg/jpeg/png.
    """
    files: List[str] = []
    exts = (".jpg", ".jpeg", ".png")
    if folders is not None:
        for f in folders:
            folder_path = os.path.join(image_root, f)
            if not os.path.isdir(folder_path):
                continue
            for name in os.listdir(folder_path):
                if name.lower().endswith(exts):
                    files.append(os.path.join(folder_path, name))
        return files

    if recursive:
        for root, _, names in os.walk(image_root):
            for name in names:
                if name.lower().endswith(exts):
                    files.append(os.path.join(root, name))
        return files
    else:
        for name in os.listdir(image_root):
            if name.lower().endswith(exts):
                files.append(os.path.join(image_root, name))
        return files


def parse_id_from_filename(path: str) -> str:
    # expects filenames like <id>_<anything>.jpg
    fname = os.path.basename(path)
    return fname.split("_")[0]


# -----------------------------
# Dataset
# -----------------------------

class ImagePriceDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, float, str]],
        transform: transforms.Compose,
        y_mean: float,
        y_std: float,
        scale_target: bool = True,
    ):
        self.samples = samples
        self.transform = transform
        self.y_mean = y_mean
        self.y_std = y_std if y_std > 0 else 1.0
        self.scale_target = scale_target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, price_eur, pid = self.samples[idx]
        with Image.open(img_path).convert("RGB") as img:
            img = self.transform(img)
        y = price_eur
        if self.scale_target:
            y = (y - self.y_mean) / self.y_std
        return img, torch.tensor([y], dtype=torch.float32), pid


# -----------------------------
# Model
# -----------------------------

def build_model(pretrained: bool = True, backbone: str = "resnet18") -> nn.Module:
    backbone = (backbone or "resnet18").lower()
    if backbone == "resnet50":
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        except Exception:
            weights = None if not pretrained else None
        model = models.resnet50(weights=weights)
    else:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        except Exception:
            weights = None if not pretrained else None
        model = models.resnet18(weights=weights)

    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_f, 1),
    )
    return model


class AttnResNetRegressor(nn.Module):
    """ResNet backbone with two heads: price regressor and attention scorer.

    Forward returns per-image price logits (unscaled) and attention scores.
    Aggregation into a bag-level prediction is done outside with softmax over attention.
    """
    def __init__(self, pretrained: bool = True, backbone: str = "resnet18"):
        super().__init__()
        bb = (backbone or "resnet18").lower()
        if bb == "resnet50":
            try:
                weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            except Exception:
                weights = None if not pretrained else None
            base = models.resnet50(weights=weights)
        else:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            except Exception:
                weights = None if not pretrained else None
            base = models.resnet18(weights=weights)
        in_f = base.fc.in_features
        # remove final fc, keep global pooling
        modules = list(base.children())[:-1]  # up to avgpool
        self.backbone = nn.Sequential(*modules)
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_f, 1),
        )
        self.attn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_f, 1),
        )
        self._feature_dim = in_f

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, C, H, W)
        feats = self.backbone(x)  # (N, feat, 1, 1)
        price = self.reg_head(feats).squeeze(1)  # (N,)
        attn = self.attn_head(feats).squeeze(1)  # (N,)
        return price, attn


# -----------------------------
# Training / Evaluation
# -----------------------------

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs(a - b) / denom) * 100.0)


def aggregate_by_id(preds: List[Tuple[str, float]]) -> Dict[str, float]:
    # preds: list of (id, pred_eur)
    by_id = defaultdict(list)
    for pid, val in preds:
        by_id[pid].append(val)
    return {k: float(np.mean(v)) for k, v in by_id.items()}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, y_mean: float, y_std: float, show_pbar: bool = False) -> Dict[str, float]:
    model.eval()
    image_level_preds: List[Tuple[str, float]] = []
    image_level_targets: List[Tuple[str, float]] = []
    with torch.no_grad():
        iterable = tqdm(loader, desc="Valid", unit="batch", leave=False) if show_pbar else loader
        for xb, yb, pid in iterable:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            # inverse-scale
            pred = out.squeeze(1).cpu().numpy() * y_std + y_mean
            tgt = yb.squeeze(1).cpu().numpy() * y_std + y_mean
            for i in range(len(pid)):
                image_level_preds.append((pid[i], float(pred[i])))
                image_level_targets.append((pid[i], float(tgt[i])))

    # aggregate to property id level
    pred_by_id = aggregate_by_id(image_level_preds)
    tgt_by_id = aggregate_by_id(image_level_targets)

    # align keys
    common_ids = sorted(set(pred_by_id.keys()) & set(tgt_by_id.keys()))
    y_true = np.array([tgt_by_id[k] for k in common_ids], dtype=np.float64)
    y_pred = np.array([pred_by_id[k] for k in common_ids], dtype=np.float64)

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "n_ids": len(common_ids),
        "n_images": len(image_level_preds),
    }


def train(
    csv_path: str = "filtered_property_data.csv",
    image_root: str = "property_images",
    out_dir: str = ".",
    batch_size: int = 32,
    epochs: int = 20,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    freeze_backbone_epochs: int = 3,
    seed: int = 42,
    device_str: str = "auto",
    target_mode: str = "price",  # "price" or "residual"
    xgb_csv: Optional[str] = None,
    backbone: str = "resnet18",
):
    set_seed(seed)

    # Device selection
    if device_str == "cpu":
        device = torch.device("cpu")
    elif device_str == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        try:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    # Ensure output directory exists (for checkpoints and stats)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load tabular targets and list all image files
    df = pd.read_csv(csv_path)
    # keep id as string to match filename parsing
    df["id"] = df["id"].astype(str)
    price_map = {row["id"]: float(row["cena"]) for _, row in df.iterrows() if not pd.isna(row["cena"])}

    # If residual mode, load XGB predictions and compute residual targets per id
    residual_map: Optional[Dict[str, float]] = None
    if target_mode == "residual":
        if not xgb_csv:
            raise ValueError("In residual target_mode, please provide --xgb_csv path with columns: id, predicted_prices")
        xgb_df = pd.read_csv(xgb_csv)
        xgb_df["id"] = xgb_df["id"].astype(str)
        # Normalize column names
        if "predicted_prices" not in xgb_df.columns and "predictions" in xgb_df.columns:
            xgb_df = xgb_df.rename(columns={"predictions": "predicted_prices"})
        if "predicted_prices" not in xgb_df.columns:
            raise ValueError("xgb_csv must contain column 'predicted_prices'")
        xgb_map = {row["id"]: float(row["predicted_prices"]) for _, row in xgb_df.iterrows()}
        residual_map = {}
        for pid, y in price_map.items():
            px = xgb_map.get(pid)
            if px is not None:
                residual_map[pid] = float(y - px)

    img_files = find_image_files(image_root)

    samples: List[Tuple[str, float, str]] = []  # (img_path, price_eur, id)
    missing = 0
    for p in img_files:
        pid = parse_id_from_filename(p)
        # choose target based on mode
        if target_mode == "residual":
            if residual_map is None:
                continue
            target_val = residual_map.get(pid)
        else:
            target_val = price_map.get(pid)

        if target_val is None or np.isnan(target_val):
            missing += 1
            continue
        samples.append((p, float(target_val), pid))

    if len(samples) == 0:
        raise RuntimeError("No samples found. Ensure images are classified into folders and CSV has matching ids.")

    # 2) Grouped split by id
    ids = np.array([s[2] for s in samples])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(np.zeros(len(samples)), groups=ids))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    # 3) Compute target scaling stats from TRAIN ONLY
    y_train = np.array([s[1] for s in train_samples], dtype=np.float64)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)

    # 4) Transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ImagePriceDataset(train_samples, train_tfms, y_mean, y_std, scale_target=True)
    val_ds = ImagePriceDataset(val_samples, val_tfms, y_mean, y_std, scale_target=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 5) Model, optimizer, loss
    model = build_model(pretrained=True, backbone=backbone).to(device)

    # Initially freeze backbone
    for name, p in model.named_parameters():
        if not name.startswith("fc"):
            p.requires_grad = False

    # Two param groups with different LRs
    params_head = [p for n, p in model.named_parameters() if n.startswith("fc") and p.requires_grad]
    params_backbone = [p for n, p in model.named_parameters() if (not n.startswith("fc")) and p.requires_grad]
    optimizer = optim.AdamW([
        {"params": params_head, "lr": lr_head},
        {"params": params_backbone, "lr": lr_backbone},
    ], weight_decay=weight_decay)

    # Huber loss tends to be robust to outliers in price
    criterion = nn.HuberLoss(delta=1.0)

    best_val_mae = float("inf")
    best_path = os.path.join(out_dir, "checkpoint_image_price.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        # Unfreeze backbone after warmup
        if epoch == (freeze_backbone_epochs + 1):
            for name, p in model.named_parameters():
                p.requires_grad = True
            # re-init optimizer to include backbone params
            optimizer = optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if n.startswith("fc")], "lr": lr_head},
                {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": lr_backbone},
            ], weight_decay=weight_decay)

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch:02d}/{epochs}", unit="batch")
        for xb, yb, _ in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, y_mean, y_std, show_pbar=True)

        print(f"Epoch {epoch:02d}/{epochs} | TrainLoss: {train_loss:.4f} | "
              f"Val MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f} | "
              f"MAPE: {val_metrics['mape']:.2f}% | n_ids: {val_metrics['n_ids']} | n_images: {val_metrics['n_images']}")

        # Early stopping on MAE
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "y_mean": y_mean,
                "y_std": y_std,
                "backbone": backbone,
            }, best_path)

    # Save training stats for inference
    stats_path = os.path.join(out_dir, "image_price_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"y_mean": y_mean, "y_std": y_std, "best_val_mae": best_val_mae, "target_mode": target_mode}, f, ensure_ascii=False, indent=2)

    print(f"Training finished. Best Val MAE: {best_val_mae:.2f}. Model saved to {best_path}")


def _build_id_to_images(image_root: str) -> Dict[str, List[str]]:
    files = find_image_files(image_root)
    id2imgs: Dict[str, List[str]] = defaultdict(list)
    for p in files:
        pid = parse_id_from_filename(p)
        id2imgs[pid].append(p)
    return id2imgs


class BagDataset(Dataset):
    """MIL bag dataset: yields (bag_images, target, id) where bag_images has shape (bag_size, C, H, W)."""
    def __init__(self,
                 ids: List[str],
                 id_to_images: Dict[str, List[str]],
                 id_to_target: Dict[str, float],
                 transform: transforms.Compose,
                 bag_size: int,
                 y_mean: float,
                 y_std: float,
                 scale_target: bool = True):
        self.ids = ids
        self.id_to_images = id_to_images
        self.id_to_target = id_to_target
        self.transform = transform
        self.bag_size = bag_size
        self.y_mean = y_mean
        self.y_std = y_std if y_std > 0 else 1.0
        self.scale_target = scale_target

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        pid = self.ids[idx]
        imgs = self.id_to_images[pid]
        # sample with replacement if not enough images
        import random
        if len(imgs) >= self.bag_size:
            chosen = random.sample(imgs, self.bag_size)
        else:
            chosen = [random.choice(imgs) for _ in range(self.bag_size)]
        tensors: List[torch.Tensor] = []
        for p in chosen:
            with Image.open(p).convert("RGB") as img:
                tensors.append(self.transform(img))
        bag = torch.stack(tensors, dim=0)  # (bag, C, H, W)
        y = float(self.id_to_target[pid])
        if self.scale_target:
            y = (y - self.y_mean) / self.y_std
        return bag, torch.tensor([y], dtype=torch.float32), pid


def _prepare_targets(csv_path: str, target_mode: str, xgb_csv: Optional[str]) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)
    price_map = {row["id"]: float(row["cena"]) for _, row in df.iterrows() if not pd.isna(row["cena"]) }
    if target_mode == "residual":
        if not xgb_csv:
            raise ValueError("In residual target_mode, please provide --xgb_csv path")
        xdf = pd.read_csv(xgb_csv)
        xdf["id"] = xdf["id"].astype(str)
        if "predicted_prices" not in xdf.columns and "predictions" in xdf.columns:
            xdf = xdf.rename(columns={"predictions": "predicted_prices"})
        if "predicted_prices" not in xdf.columns:
            raise ValueError("xgb_csv must contain column 'predicted_prices'")
        xmap = {row["id"]: float(row["predicted_prices"]) for _, row in xdf.iterrows()}
        return {pid: float(price_map[pid] - xmap[pid]) for pid in price_map.keys() if pid in xmap}
    return price_map


def train_mil(
    csv_path: str = "filtered_property_data.csv",
    image_root: str = "property_images",
    out_dir: str = ".",
    batch_size: int = 8,  # number of bags per batch
    bag_size: int = 8,
    epochs: int = 15,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device_str: str = "auto",
    target_mode: str = "price",
    xgb_csv: Optional[str] = None,
    backbone: str = "resnet18",
):
    set_seed(seed)

    # Device selection
    if device_str == "cpu":
        device = torch.device("cpu")
    elif device_str == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)

    # Build index and targets
    id_to_images = _build_id_to_images(image_root)
    id_to_target = _prepare_targets(csv_path, target_mode, xgb_csv)
    # keep only ids with images and targets
    ids = sorted(list(set(id_to_images.keys()) & set(id_to_target.keys())))
    if not ids:
        raise RuntimeError("No overlapping ids with images and targets found")

    # Split by property id
    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=seed)

    # Compute scaling on train targets only
    y_train = np.array([id_to_target[i] for i in train_ids], dtype=np.float64)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)

    # Transforms
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = BagDataset(train_ids, id_to_images, id_to_target, train_tfms, bag_size, y_mean, y_std, scale_target=True)
    val_ds = BagDataset(val_ids, id_to_images, id_to_target, val_tfms, bag_size, y_mean, y_std, scale_target=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = AttnResNetRegressor(pretrained=True, backbone=backbone).to(device)

    # Freeze backbone at start
    for p in model.backbone.parameters():
        p.requires_grad = False

    params_head = list(model.reg_head.parameters()) + list(model.attn_head.parameters())
    optimizer = optim.AdamW([
        {"params": params_head, "lr": lr_head},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": lr_backbone},
    ], weight_decay=weight_decay)

    criterion = nn.HuberLoss(delta=1.0)
    best_val_mae = float("inf")
    best_path = os.path.join(out_dir, "checkpoint_image_price.pth")

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after 3 epochs
        if epoch == 4:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {"params": list(model.reg_head.parameters()) + list(model.attn_head.parameters()), "lr": lr_head},
                {"params": model.backbone.parameters(), "lr": lr_backbone},
            ], weight_decay=weight_decay)

        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"TrainMIL {epoch:02d}/{epochs}", unit="bag")
        for xb, yb, _ in pbar:
            # xb: (B, bag, C, H, W)
            B, K = xb.shape[0], xb.shape[1]
            xb = xb.view(B * K, *xb.shape[2:]).to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).squeeze(1)  # (B,)
            optimizer.zero_grad(set_to_none=True)
            price_i, attn_i = model(xb)  # (B*K,), (B*K,)
            price_i = price_i.view(B, K)
            attn_i = attn_i.view(B, K)
            w = torch.softmax(attn_i, dim=1)
            bag_pred = (w * price_i).sum(dim=1)  # (B,)
            loss = criterion(bag_pred.unsqueeze(1), yb.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running += loss.item() * B
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / len(train_loader.dataset)

        # Validation
        model.eval()
        with torch.no_grad():
            image_level_preds: List[Tuple[str, float, float]] = []  # (id, price, attn)
            image_level_targets: Dict[str, float] = {}
            for xb, yb, pid in tqdm(val_loader, desc="ValidMIL", unit="bag", leave=False):
                B, K = xb.shape[0], xb.shape[1]
                xb = xb.view(B * K, *xb.shape[2:]).to(device)
                yb = yb.to(device).squeeze(1)
                price_i, attn_i = model(xb)
                price_i = price_i.view(B, K).cpu().numpy()
                attn_i = attn_i.view(B, K).cpu().numpy()
                for i in range(B):
                    # softmax per bag
                    wi = np.exp(attn_i[i] - np.max(attn_i[i]))
                    wi = wi / np.sum(wi)
                    pred = float(np.sum(wi * price_i[i]) * y_std + y_mean)
                    image_level_preds.append((pid[i], pred, 1.0))  # store one per bag/id
                    # store target once per id
                    if pid[i] not in image_level_targets:
                        image_level_targets[pid[i]] = float(yb[i].item() * y_std + y_mean)

        # Aggregate by id (already one per id per batch, but ids may repeat less likely)
        pred_by_id = defaultdict(list)
        for pid, val, _ in image_level_preds:
            pred_by_id[pid].append(val)
        tgt_by_id = image_level_targets
        common = sorted(set(pred_by_id.keys()) & set(tgt_by_id.keys()))
        y_true = np.array([tgt_by_id[k] for k in common], dtype=np.float64)
        y_pred = np.array([np.mean(pred_by_id[k]) for k in common], dtype=np.float64)

        val_mae = mae(y_true, y_pred)
        val_rmse = rmse(y_true, y_pred)
        val_mape = mape(y_true, y_pred)
        print(f"Epoch {epoch:02d}/{epochs} | TrainLoss: {train_loss:.4f} | Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | MAPE: {val_mape:.2f}% | n_ids: {len(common)}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "y_mean": y_mean,
                "y_std": y_std,
                "backbone": backbone,
                "mil": True,
            }, best_path)

    with open(os.path.join(out_dir, "image_price_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"y_mean": y_mean, "y_std": y_std, "best_val_mae": best_val_mae, "target_mode": target_mode, "mil": True}, f, ensure_ascii=False, indent=2)

    print(f"Training MIL finished. Best Val MAE: {best_val_mae:.2f}. Model saved to {best_path}")


def predict_folder(
    image_root: str,
    checkpoint_path: str,
    out_csv: str = "image_price_predictions.csv",
    target_mode: str = "price",
    xgb_csv: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    y_mean = ckpt.get("y_mean", 0.0)
    y_std = ckpt.get("y_std", 1.0) or 1.0
    ckpt_target_mode = ckpt.get("target_mode", None)
    if ckpt_target_mode and target_mode != ckpt_target_mode:
        print(f"Warning: checkpoint trained for target_mode={ckpt_target_mode}, but predict called with target_mode={target_mode}")

    ckpt_backbone = ckpt.get("backbone", "resnet18")
    is_mil = bool(ckpt.get("mil", False))
    if is_mil:
        model = AttnResNetRegressor(pretrained=False, backbone=ckpt_backbone)
    else:
        model = build_model(pretrained=False, backbone=ckpt_backbone)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # If residual predictions, we need XGB predictions to add back
    xgb_map: Dict[str, float] = {}
    if target_mode == "residual":
        if not xgb_csv:
            raise ValueError("In residual predict mode, provide --xgb_csv to combine residual with XGB predicted price")
        xdf = pd.read_csv(xgb_csv)
        xdf["id"] = xdf["id"].astype(str)
        if "predicted_prices" not in xdf.columns and "predictions" in xdf.columns:
            xdf = xdf.rename(columns={"predictions": "predicted_prices"})
        if "predicted_prices" not in xdf.columns:
            raise ValueError("xgb_csv must contain column 'predicted_prices'")
        xgb_map = {row["id"]: float(row["predicted_prices"]) for _, row in xdf.iterrows()}
        print(f"Residual mode: loaded XGB predictions for {len(xgb_map)} ids")

    id_to_images = _build_id_to_images(image_root)
    all_imgs = sum((len(v) for v in id_to_images.values()), 0)
    print(f"Found {all_imgs} images under '{image_root}' across {len(id_to_images)} properties")
    agg_rows = []
    skipped_no_base = 0

    with torch.no_grad():
        for pid, paths in tqdm(list(id_to_images.items()), desc="PredictIDs", unit="id"):
            per_img_preds: List[float] = []
            per_img_attn: List[float] = []
            for p in paths:
                with Image.open(p).convert("RGB") as img:
                    xb = val_tfms(img).unsqueeze(0).to(device)
                if is_mil:
                    price_i, attn_i = model(xb)
                    pred_val = float(price_i.item() * y_std + y_mean)
                    score = float(attn_i.item())
                else:
                    out = model(xb)
                    pred_val = float(out.item() * y_std + y_mean)
                    score = 0.0
                per_img_preds.append(pred_val)
                per_img_attn.append(score)

            per_img_preds = np.array(per_img_preds, dtype=np.float64)
            if target_mode == "residual":
                base = xgb_map.get(pid)
                if base is None:
                    skipped_no_base += len(paths)
                    continue
                per_img_preds = per_img_preds + float(base)

            if is_mil and len(per_img_attn) > 0:
                a = np.array(per_img_attn, dtype=np.float64)
                a = a - np.max(a)
                w = np.exp(a)
                w = w / np.sum(w)
                mean = float(np.sum(w * per_img_preds))
                var = float(np.sum(w * (per_img_preds - mean) ** 2))
                std = float(np.sqrt(max(var, 0.0)))
            else:
                mean = float(np.mean(per_img_preds))
                std = float(np.std(per_img_preds)) if len(per_img_preds) > 1 else 0.0

            agg_rows.append({
                "id": pid,
                "pred_price_eur": mean,
                "pred_std_eur": std,
                "n_images": int(len(paths)),
            })

    pd.DataFrame(agg_rows).to_csv(out_csv, index=False)
    if target_mode == "residual":
        print(f"Skipped {skipped_no_base} images due to missing XGB base prediction")
    print(f"Wrote aggregated predictions to {out_csv} ({len(agg_rows)} properties)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or run image-only price model")
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--csv_path", type=str, default="filtered_property_data.csv")
    p_train.add_argument("--image_root", type=str, default="property_images")
    p_train.add_argument("--out_dir", type=str, default=".")
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr_backbone", type=float, default=1e-4)
    p_train.add_argument("--lr_head", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--freeze_backbone_epochs", type=int, default=3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p_train.add_argument("--target_mode", type=str, choices=["price", "residual"], default="price")
    p_train.add_argument("--xgb_csv", type=str, default=None, help="Required for residual mode: xgb_predictions.csv path")
    p_train.add_argument("--backbone", type=str, choices=["resnet18", "resnet50"], default="resnet18")

    p_train_mil = subparsers.add_parser("train_mil")
    p_train_mil.add_argument("--csv_path", type=str, default="filtered_property_data.csv")
    p_train_mil.add_argument("--image_root", type=str, default="property_images")
    p_train_mil.add_argument("--out_dir", type=str, default=".")
    p_train_mil.add_argument("--batch_size", type=int, default=8, help="number of bags per batch")
    p_train_mil.add_argument("--bag_size", type=int, default=8)
    p_train_mil.add_argument("--epochs", type=int, default=15)
    p_train_mil.add_argument("--lr_backbone", type=float, default=1e-4)
    p_train_mil.add_argument("--lr_head", type=float, default=1e-3)
    p_train_mil.add_argument("--weight_decay", type=float, default=1e-4)
    p_train_mil.add_argument("--seed", type=int, default=42)
    p_train_mil.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p_train_mil.add_argument("--target_mode", type=str, choices=["price", "residual"], default="price")
    p_train_mil.add_argument("--xgb_csv", type=str, default=None)
    p_train_mil.add_argument("--backbone", type=str, choices=["resnet18", "resnet50"], default="resnet18")

    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--image_root", type=str, default="property_images")
    p_pred.add_argument("--checkpoint", type=str, default="checkpoint_image_price.pth")
    p_pred.add_argument("--out_csv", type=str, default="image_price_predictions.csv")
    p_pred.add_argument("--target_mode", type=str, choices=["price", "residual"], default="price")
    p_pred.add_argument("--xgb_csv", type=str, default=None, help="Required for residual mode: xgb_predictions.csv path")

    args = parser.parse_args()

    if args.cmd == "predict":
        predict_folder(args.image_root, args.checkpoint, args.out_csv, target_mode=args.target_mode, xgb_csv=args.xgb_csv)
    elif args.cmd == "train_mil":
        train_mil(
            csv_path=args.csv_path,
            image_root=args.image_root,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            bag_size=args.bag_size,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device_str=args.device,
            target_mode=args.target_mode,
            xgb_csv=args.xgb_csv,
            backbone=args.backbone,
        )
    else:
        train(
            csv_path=args.csv_path,
            image_root=args.image_root,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            weight_decay=args.weight_decay,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            seed=args.seed,
            device_str=args.device,
            target_mode=args.target_mode,
            xgb_csv=args.xgb_csv,
            backbone=args.backbone,
        )
