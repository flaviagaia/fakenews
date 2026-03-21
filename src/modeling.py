from collections import Counter
from pathlib import Path

import joblib
import torch
from torch import nn
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(texts, max_features=10000):
    words = " ".join(texts).lower().split()
    vocab = {word: index + 2 for index, (word, _) in enumerate(Counter(words).most_common(max_features))}
    vocab[PAD_TOKEN] = 0
    vocab[UNK_TOKEN] = 1
    return vocab


def encode_text(text, vocab, max_len=100):
    token_ids = [vocab.get(word, vocab[UNK_TOKEN]) for word in text.lower().split()[:max_len]]
    return token_ids + [vocab[PAD_TOKEN]] * (max_len - len(token_ids))


class NewsDataset(Dataset):
    def __init__(self, frame, vocab, max_len=100, is_test=False):
        self.inputs = [encode_text(text, vocab, max_len=max_len) for text in frame["text"]]
        self.targets = frame["category"].values if not is_test else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        features = torch.tensor(self.inputs[index], dtype=torch.long)
        if self.targets is None:
            return features
        return features, torch.tensor(self.targets[index], dtype=torch.float32)


class FakeNewsGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, hidden_state = self.gru(self.embedding(x))
        return torch.sigmoid(self.classifier(hidden_state[0]))


def save_artifacts(model, vocab, path):
    payload = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "model_config": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
        },
    }
    torch.save(payload, path)


def load_artifacts(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    vocab = payload["vocab"]
    config = payload.get("model_config", {})
    model = FakeNewsGRU(
        vocab_size=len(vocab),
        embedding_dim=config.get("embedding_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab


def save_metrics(metrics, path):
    joblib.dump(metrics, path)


def load_metrics(path):
    return joblib.load(path)
