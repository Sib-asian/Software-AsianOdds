"""
Train Odds Movement LSTM
========================

CLI per addestrare il modello LSTM usato dal BLOCCO 6 sfruttando la history
salvata (cache/odds_history). Usa PyTorch e salva pesi + scaler su models/odds_lstm.pth.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from ..config import AIConfig
from ..blocco_6_odds_tracker import OddsLSTM, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    import torch.nn as nn
else:
    raise ImportError("PyTorch è richiesto per addestrare l'odds tracker (pip install torch)")

logger = logging.getLogger(__name__)


def _extract_price(entry: dict, selection: str) -> float | None:
    if "odds" in entry:
        return entry.get("odds")
    prices = entry.get("prices") or {}
    price_entry = prices.get(selection)
    if isinstance(price_entry, dict):
        return price_entry.get("price")
    return None


def _load_single_history(path: Path, selection: str, min_points: int) -> List[float]:
    try:
        with path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Skipping %s (%s)", path, exc)
        return []

    if not isinstance(entries, list):
        return []

    def _timestamp(entry: dict) -> str:
        return entry.get("timestamp") or entry.get("time") or ""

    entries = sorted(entries, key=_timestamp)
    prices: List[float] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        price = _extract_price(entry, selection)
        if isinstance(price, (int, float)):
            prices.append(float(price))

    if len(prices) < min_points:
        return []
    return prices


def build_sequences(
    history_dir: Path,
    selection: str,
    sequence_length: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    sequences: List[List[float]] = []
    targets: List[float] = []
    files = sorted(history_dir.glob("*.json"))

    for path in files:
        series = _load_single_history(path, selection, sequence_length + horizon)
        if not series:
            continue
        for idx in range(len(series) - sequence_length - horizon + 1):
            window = series[idx:idx + sequence_length]
            target = series[idx + sequence_length + horizon - 1]
            sequences.append(window)
            targets.append(target)

    if not sequences:
        raise RuntimeError(f"Nessuna sequenza valida trovata in {history_dir}")

    seq_arr = np.array(sequences, dtype=np.float32)
    tgt_arr = np.array(targets, dtype=np.float32)
    return seq_arr, tgt_arr


class OddsHistoryDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences[:, :, None], dtype=torch.float32)
        self.targets = torch.tensor(targets[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.targets[idx]


def train_model(
    dataset: Dataset,
    sequence_length: int,
    scaler: dict,
    config: AIConfig,
    epochs: int,
    batch_size: int,
    device: str
) -> OddsLSTM:
    hidden_size = config.odds_lstm_units
    num_layers = config.odds_lstm_layers
    model = OddsLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, targets in train_loader:
            seqs = seqs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(seqs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs = seqs.to(device)
                targets = targets.to(device)
                preds = model(seqs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * len(seqs)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        logger.info("Epoch %03d | train=%.5f | val=%.5f", epoch, train_loss, val_loss)

    payload = {
        "state_dict": model.state_dict(),
        "sequence_length": sequence_length,
        "selection": scaler.get("selection", "home"),
        "scaler": {"mean": scaler["mean"], "std": scaler["std"]},
    }
    torch.save(payload, Path(config.models_dir) / "odds_lstm.pth")
    logger.info("✅ Modello salvato in %s", Path(config.models_dir) / "odds_lstm.pth")
    return model


def run_training(
    history_dir: Path,
    selection: str = "home",
    sequence_length: Optional[int] = None,
    horizon: int = 1,
    epochs: int = 30,
    batch_size: int = 64,
    device: str = "cpu",
    config: Optional[AIConfig] = None,
) -> Dict[str, Any]:
    config = config or AIConfig()
    history_dir = Path(history_dir)
    if not history_dir.exists():
        raise FileNotFoundError(f"Directory odds history inesistente: {history_dir}")

    sequence_len = sequence_length or config.odds_lookback_window
    logger.info("Carico sequences da %s (selection=%s, window=%d)", history_dir, selection, sequence_len)
    sequences, targets = build_sequences(history_dir, selection, sequence_len, horizon)

    values = np.concatenate([sequences.flatten(), targets])
    mean = float(values.mean())
    std = float(values.std()) or 1.0
    sequences = (sequences - mean) / std
    targets = (targets - mean) / std

    dataset = OddsHistoryDataset(sequences, targets)
    logger.info("Dataset pronto: %d samples", len(dataset))

    scaler = {"mean": mean, "std": std, "selection": selection}
    train_model(
        dataset=dataset,
        sequence_length=sequence_len,
        scaler=scaler,
        config=config,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    return {
        "samples": len(dataset),
        "sequence_length": sequence_len,
        "selection": selection,
        "history_dir": str(history_dir),
        "model_path": str(Path(config.models_dir) / "odds_lstm.pth"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Odds LSTM using cached odds history.")
    parser.add_argument("--history-dir", type=str, help="Directory con gli odds history JSON.")
    parser.add_argument("--models-dir", type=str, help="Cartella output modelli.")
    parser.add_argument("--selection", type=str, default="home", help="Selezione da prevedere (home/away/draw).")
    parser.add_argument("--sequence-length", type=int, default=None, help="Finestre storiche (default: config.odds_lookback_window).")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in passi (default: 1).")
    parser.add_argument("--epochs", type=int, default=30, help="Numero di epoche.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = AIConfig()
    if args.models_dir:
        config.models_dir = Path(args.models_dir)
        config.models_dir.mkdir(parents=True, exist_ok=True)
    history_dir = Path(args.history_dir or (config.cache_dir / "odds_history"))

    result = run_training(
        history_dir=history_dir,
        selection=args.selection,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        config=config,
    )
    logger.info("Training completato: %s", result)


if __name__ == "__main__":
    main()
