from src.predict import generate_submission
from src.train import train_model


def run_demo():
    training_result = train_model()
    submission, output_path = generate_submission(model_path=training_result["model_path"])

    print("Fake News Classification Assistant")
    print("-" * 40)
    print(f"Validation F1-score: {training_result['metrics']['f1_score']:.2f}")
    print(f"Predictions generated: {len(submission)}")
    print("Preview:")
    print(submission.head().to_string(index=False))
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    run_demo()
