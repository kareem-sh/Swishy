"""Isolated official KaliCalib challenge-model inference for one image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from config.settings import PROJECT_ROOT
from court.calibration import CourtCalibrator, draw_calibration_debug, get_field_points
from court.models import CourtDetection, CourtKeypoint
from court.swishy_calibration import adapt_from_config
from utils.config_loader import load_yaml


def resolve_device(preferred: str | None = "auto") -> str:
    """Return CUDA when available, otherwise a safe CPU device."""
    import torch

    choice = (preferred or "auto").lower()
    if choice == "cpu":
        return "cpu"
    if choice in {"auto", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError("court.device must be one of: auto, cuda, cpu")


class CourtDetector:
    """Load KaliCalib once and turn a BGR image into court keypoints."""

    _models: dict[tuple[Path, str], Any] = {}

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        input_width: int | None = None,
        input_height: int | None = None,
        model_factory: Callable[[], Any] | None = None,
    ):
        cfg = load_yaml("court.yaml")
        requested_path = model_path or cfg.get("model_path", "models/court/model_challenge.pth")
        self.model_path = validate_model_path(requested_path)
        self.device = resolve_device(device or cfg.get("device", "auto"))
        self.input_width = int(input_width or cfg.get("input_width", 960))
        self.input_height = int(input_height or cfg.get("input_height", 540))
        if (self.input_width, self.input_height) != (960, 540):
            raise ValueError("KaliCalib challenge weights require a 960x540 input image.")
        self._model = self._load_model(model_factory)
        self._field_points = get_field_points()

    def _load_model(self, model_factory: Callable[[], Any] | None) -> Any:
        if model_factory is not None:
            return self._load_weights(model_factory())
        key = (self.model_path, self.device)
        if key not in self._models:
            from third_party.kalicalib.model_resnet import makeModel

            self._models[key] = self._load_weights(makeModel())
        return self._models[key]

    def _load_weights(self, model: Any) -> Any:
        import torch

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        # DataParallel checkpoints are common; official weights are otherwise
        # loaded unchanged through makeModel/load_state_dict.
        if isinstance(state, dict) and any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
        return model.to(self.device).eval()

    def detect(self, frame: np.ndarray | None) -> CourtDetection:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return CourtDetection(0, 0, status="invalid_image", message="Expected a non-empty BGR image with three channels.")
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return CourtDetection(width, height, status="invalid_image", message="Image has no pixels.")
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        heatmaps = self._infer(resized)
        keypoints = self._decode_heatmaps(heatmaps, width, height)
        confidence = float(np.mean([point.confidence for point in keypoints])) if keypoints else None
        return CourtDetection(
            image_width=width,
            image_height=height,
            keypoints=keypoints,
            valid_keypoints=list(keypoints),
            model_confidence=confidence,
            status="ok",
            debug={"input_size": [self.input_width, self.input_height], "heatmap_shape": list(heatmaps.shape)},
        )

    def _infer(self, resized_bgr: np.ndarray) -> np.ndarray:
        import torch

        # The official path uses ToTensor + ImageNet normalization. OpenCV
        # provides BGR frames, so convert to RGB before that exact normalization.
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        with torch.no_grad():
            output = self._model(tensor)
        return output[0].detach().cpu().numpy()

    def _decode_heatmaps(self, heatmaps: np.ndarray, image_width: int, image_height: int) -> list[CourtKeypoint]:
        if heatmaps.ndim != 3 or heatmaps.shape[0] < 94:
            raise ValueError("Unexpected KaliCalib output; expected 94 heatmap channels.")
        # Official estimateCalibHM uses channels 0..90 (94 - 1 - 2), finds
        # each channel's maximum, and maps the quarter-resolution coordinate
        # back to the 960x540 inference image. We then map it to the source.
        scale_x = image_width / float(self.input_width)
        scale_y = image_height / float(self.input_height)
        output_scale_x = self.input_width / float(heatmaps.shape[2])
        output_scale_y = self.input_height / float(heatmaps.shape[1])
        keypoints: list[CourtKeypoint] = []
        for index in range(91):
            channel = heatmaps[index]
            y, x = np.unravel_index(int(np.argmax(channel)), channel.shape)
            confidence = float(channel[y, x])
            xy = ((x + 0.5) * output_scale_x * scale_x, (y + 0.5) * output_scale_y * scale_y)
            court_xy = tuple(float(value) for value in self._field_points[index, :2])
            keypoints.append(CourtKeypoint(index, xy, court_xy, confidence))
        return keypoints


def validate_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"KaliCalib model weights were not found: {path}")
    return path


def _parse_pixel(value: str) -> tuple[float, float]:
    try:
        x, y = value.split(",", 1)
        return float(x), float(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Pixels must be provided as u,v (for example: 500,600).") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated KaliCalib court calibration on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    parser.add_argument("--output", type=Path, help="Write keypoint/court-outline visualization here.")
    parser.add_argument("--print-court-coordinates", action="store_true", help="Print configured/manual image-floor-pixel diagnostics.")
    parser.add_argument("--point", action="append", type=_parse_pixel, default=[], metavar="U,V", help="Add a floor pixel diagnostic; may be repeated.")
    args = parser.parse_args()
    frame = cv2.imread(str(args.image))
    if frame is None:
        parser.error(f"Could not read image: {args.image}")
    detector = CourtDetector(model_path=args.model, device=args.device)
    detection = detector.detect(frame)
    calibration = CourtCalibrator().calibrate(frame, detection)
    swishy = adapt_from_config(calibration)
    cfg = load_yaml("court.yaml")
    configured_points = cfg.get("diagnostic_points_px", [])
    diagnostic_points = list(configured_points) + list(args.point)
    tolerance = float(cfg.get("sanity", {}).get("round_trip_tolerance_px", 1.0))
    diagnostics = swishy.diagnose_image_points(diagnostic_points, tolerance) if diagnostic_points else []
    print(f"device: {detector.device}")
    print(f"detected keypoints: {len(detection.keypoints)}")
    print(f"Inliers: {calibration.inlier_count}")
    print(f"Calibration: {calibration.status.upper() if calibration.status else 'INVALID'}")
    print(f"Reprojection error: {calibration.reprojection_error_px} px")
    print(f"Reference hoop: {swishy.calibration.court_frame.reference_hoop}")
    print(f"Court: {swishy.calibration.sanity.estimated_court_length_m:.2f}m x {swishy.calibration.sanity.estimated_court_width_m:.2f}m")
    print(f"dimension error: length={swishy.calibration.sanity.length_error_m:.3f}m width={swishy.calibration.sanity.width_error_m:.3f}m")
    print(f"orientation: +X court-right, +Y away from {swishy.calibration.court_frame.reference_hoop} hoop, +Z up")
    print(f"swishy court frame: {swishy.calibration.status}")
    if args.print_court_coordinates:
        if not diagnostics:
            print("No diagnostic points configured. Add --point U,V or court.diagnostic_points_px.")
        for index, item in enumerate(diagnostics, 1):
            print(f"Test point {index}:")
            print(f" image = ({item.image_xy[0]:.3f}, {item.image_xy[1]:.3f})")
            if item.court_point is not None:
                point = item.court_point
                print(f" court = ({point.x_m:.3f}, {point.y_m:.3f}, {point.z_m:.3f}) m")
                print(f" projected_back = ({item.projected_back_xy[0]:.3f}, {item.projected_back_xy[1]:.3f})")
                print(f" round_trip_error = {item.round_trip_error_px:.6f} px ({item.status})")
            else:
                print(f" status = {item.status}: {item.message}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), draw_calibration_debug(frame, calibration, swishy, diagnostics)):
            raise RuntimeError(f"Could not write visualization: {args.output}")
        print(f"visualization: {args.output}")
    return 0 if calibration.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
