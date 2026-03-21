import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import generate_submission  # noqa: E402
from src.train import train_model  # noqa: E402


class FakeNewsPipelineTest(unittest.TestCase):
    def test_training_and_submission_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.joblib"
            submission_path = Path(temp_dir) / "submissions.csv"

            training_result = train_model(model_output=model_path, seed=7)
            submission, saved_path = generate_submission(
                model_path=model_path,
                output_file=submission_path,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(saved_path.exists())
            self.assertEqual(list(submission.columns), ["id", "category"])
            self.assertEqual(len(submission), 10)
            self.assertIn("f1_score", training_result["metrics"])


if __name__ == "__main__":
    unittest.main()
