from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_utils import combine_text_columns, data_path, load_csv
from .modeling import NewsDataset, load_artifacts


def generate_submission(
    model_path=None,
    test_file=None,
    output_file=None,
    max_len=100,
):
    model_path = Path(model_path or Path(__file__).resolve().parents[1] / "artifacts" / "fake_news_gru.pt")
    test_file = Path(test_file or data_path("test.csv"))
    output_file = Path(output_file or Path(__file__).resolve().parents[1] / "artifacts" / "submissions.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_df = combine_text_columns(load_csv(test_file))
    model, vocab = load_artifacts(model_path, device=device)
    test_loader = DataLoader(NewsDataset(test_df, vocab, max_len=max_len, is_test=True), batch_size=32)

    predictions = []
    with torch.no_grad():
        for features in test_loader:
            probs = (model(features.to(device)) > 0.5).int().cpu().numpy()
            predictions.extend(probs.flatten())

    submission = pd.DataFrame({"id": test_df["id"], "category": predictions})
    submission.to_csv(output_file, index=False)
    return submission, output_file


if __name__ == "__main__":
    _, output_path = generate_submission()
    print(f"Submission saved to: {output_path}")
