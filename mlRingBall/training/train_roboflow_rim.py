from pathlib import Path

from ultralytics import YOLO

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent

DATASET_YAML = (
    ML_ROOT
    / "datasets"
    / "basketball_hoop_images"
    / "data.yaml"
)

STARTING_MODEL = (
    PROJECT_ROOT
    / "models"
    / "BODD_yolov8n_0001.pt"
)


def main() -> None:
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML was not found: {DATASET_YAML}"
        )

    if not STARTING_MODEL.is_file():
        raise FileNotFoundError(
            f"Starting model was not found: {STARTING_MODEL}"
        )

    # The existing E-BARD model already understands basketball and hoop.
    model = YOLO(str(STARTING_MODEL), task="detect")

    results = model.train(
        data=str(DATASET_YAML),

        epochs=150,
        patience=30,
        imgsz=960,

        batch=4,
        device=0,
        workers=4,
        amp=True,
        cache="disk",

        # Color and lighting augmentation
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,

        # Camera and object-position augmentation
        degrees=5,
        translate=0.10,
        scale=0.40,
        perspective=0.0005,
        fliplr=0.5,

        mosaic=0.5,
        close_mosaic=15,

        project=str(ML_ROOT / "runs" / "basketball_rim_training"),
        name="basketball_and_rim",
        plots=True,
        save_period=10,
    )

    best_model = results.save_dir / "weights" / "best.pt"

    # Verify the trained class names.
    trained_model = YOLO(str(best_model), task="detect")

    print(f"Best model saved at: {best_model}")
    print(f"Trained classes: {trained_model.names}")


if __name__ == "__main__":
    main()
