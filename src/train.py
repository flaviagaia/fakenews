import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_utils import combine_text_columns, data_path, load_csv
from .modeling import FakeNewsGRU, NewsDataset, build_vocab, save_artifacts, save_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    train_file=None,
    model_output=None,
    metrics_output=None,
    max_len=100,
    batch_size=1,
    epochs=5,
    embedding_dim=64,
    hidden_dim=32,
    learning_rate=0.01,
    seed=42,
):
    set_seed(seed)
    train_file = Path(train_file or data_path("train.csv"))
    model_output = Path(model_output or Path(__file__).resolve().parents[1] / "artifacts" / "fake_news_gru.pt")
    metrics_output = Path(metrics_output or Path(__file__).resolve().parents[1] / "artifacts" / "metrics.joblib")
    model_output.parent.mkdir(parents=True, exist_ok=True)

    train_df = combine_text_columns(load_csv(train_file))

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["category"],
    )
    vocab = build_vocab(train_split["text"], max_features=10000)

    train_loader = DataLoader(NewsDataset(train_split, vocab, max_len=max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NewsDataset(val_split, vocab, max_len=max_len), batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FakeNewsGRU(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features.to(device))
            loss = criterion(outputs, targets.to(device).view(-1, 1))
            loss.backward()
            optimizer.step()

    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            probs = (model(features.to(device)) > 0.5).cpu().numpy()
            predictions.extend(probs.flatten())
            truths.extend(targets.numpy())

    f1 = f1_score(truths, predictions)
    metrics = {
        "f1_score": float(f1),
        "validation_size": len(val_split),
        "device": device,
        "batch_size": batch_size,
        "epochs": epochs,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "stratified_split": True,
        "learning_rate": learning_rate,
    }

    save_artifacts(model, vocab, model_output)
    save_metrics(metrics, metrics_output)
    return {"model_path": str(model_output), "metrics_path": str(metrics_output), "metrics": metrics}


if __name__ == "__main__":
    result = train_model()
    print("Fake News Classification Assistant")
    print("-" * 40)
    print(f"Validation F1-score: {result['metrics']['f1_score']:.2f}")
    print(f"Validation size: {result['metrics']['validation_size']}")
    print(f"Device: {result['metrics']['device']}")
    print(f"Model saved to: {result['model_path']}")
