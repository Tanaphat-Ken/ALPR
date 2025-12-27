"""Utilities for preparing license-plate OCR datasets.

This module covers a few key steps required by the PoC plan:

* Load the provided CSV export and keep only validated rows (`is_validate == True`).
* Resolve relative image paths (both context image and plate crop) against one
  or more candidate data roots.
* Parse bounding boxes stored as stringified lists so that downstream scripts
  can crop plates or characters if needed.
* Produce stratified train/validation/test splits with deterministic shuffling.
* Export lightweight manifests (JSON/JSONL) that are compatible with
  HuggingFace Datasets or custom PyTorch dataloaders.

All helpers operate on `LicensePlateRecord`, a dataclass that keeps rich
metadata for each sample while maintaining simple accessors for the training
pipeline (e.g. `plate_image_path` + `plate_text`).
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
	from PIL import Image
except ImportError:  # pragma: no cover - optional dependency for crop helpers
	Image = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - hints for type checkers only
	from PIL import Image as PILImage
else:
	PILImage = Any  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LicensePlateRecord:
	"""Represents a single validated license-plate sample."""

	index: int
	plate_text: str
	plate_image_path: Path
	raw_image_path: Optional[Path]
	cameras_plate_no: Optional[str] = None
	province_code: Optional[str] = None
	province_description: Optional[str] = None
	is_validate: bool = True
	car_bbox: Optional[List[float]] = None
	plate_bbox: Optional[List[float]] = None
	character_bboxes: Optional[List[List[List[float]]]] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_example(self, include_metadata: bool = False) -> Dict[str, Any]:
		"""Return a minimal dictionary for training/evaluation datasets."""

		example = {
			"image_path": str(self.plate_image_path),
			"text": self.plate_text,
			"province_code": self.province_code,
			"province_description": self.province_description,
			"cameras_plate_no": self.cameras_plate_no,
		}
		if include_metadata:
			example["metadata"] = self.metadata.copy()
		return example


def _safe_literal_eval(value: str | None) -> Any:
	if value is None:
		return None
	value = value.strip()
	if not value:
		return None
	try:
		return ast.literal_eval(value)
	except (ValueError, SyntaxError):
		logger.warning("Failed to parse literal: %s", value[:80])
		return None


def _ensure_path_candidates(data_roots: Optional[Sequence[str | Path]], csv_path: Path) -> List[Path]:
	candidates: List[Path] = []
	if data_roots:
		candidates.extend(Path(p).resolve() for p in data_roots)
	# fallbacks: CSV directory and its parent (covers ./data & repo root cases)
	candidates.append(csv_path.parent.resolve())
	candidates.append(csv_path.parent.parent.resolve())
	# Deduplicate while preserving order
	seen: set[Path] = set()
	unique: List[Path] = []
	for cand in candidates:
		if cand not in seen:
			unique.append(cand)
			seen.add(cand)
	return unique


def _resolve_image_path(relative_path: str, data_roots: Sequence[Path]) -> Optional[Path]:
	rel = relative_path.replace("\\", "/").lstrip("/")
	for root in data_roots:
		candidate = root / rel
		if candidate.exists():
			return candidate
		# Some datasets store images under an extra "data" folder; try injecting it
		alt = root / "data" / rel
		if alt.exists():
			return alt
	logger.debug("Unable to resolve image path '%s'", relative_path)
	return None


def load_records(
	csv_path: str | Path,
	*,
	data_roots: Optional[Sequence[str | Path]] = None,
	require_validate: bool = True,
	drop_missing_images: bool = True,
) -> List[LicensePlateRecord]:
	"""Load validated records from the CSV export.

	Parameters
	----------
	csv_path:
		Path to the CSV file (tb_match_data_....csv).
	data_roots:
		Candidate root directories containing the `/210/...` image hierarchy. If
		omitted, the CSV folder and its parent will be used automatically.
	require_validate:
		Keep only rows where `is_validate` resolves to True.
	drop_missing_images:
		Skip rows whose plate crop file cannot be resolved. Set to False if you
		would rather keep them (e.g. to diagnose missing files).
	"""

	csv_path = Path(csv_path).resolve()
	roots = _ensure_path_candidates(data_roots, csv_path)
	records: List[LicensePlateRecord] = []

	with csv_path.open(newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for idx, row in enumerate(reader):
			is_validate_raw = row.get("is_validate")
			is_validate_flag = str(is_validate_raw).strip().lower() in {"true", "1", "yes"}
			if require_validate and not is_validate_flag:
				continue

			plate_text = (row.get("plate") or row.get("cameras_plateNo1") or "").strip()
			if not plate_text:
				logger.debug("Skipping row %s without plate text", idx)
				continue

			plate_rel = row.get("image_name_gray") or row.get("image_name")
			if not plate_rel:
				logger.debug("Skipping row %s without image path", idx)
				continue

			plate_path = _resolve_image_path(plate_rel, roots)
			if not plate_path:
				if drop_missing_images:
					logger.warning("Missing plate image for row %s: %s", idx, plate_rel)
					continue
				logger.debug("Keeping unresolved image for row %s: %s", idx, plate_rel)

			raw_path = None
			raw_rel = row.get("image_name")
			if raw_rel:
				raw_path = _resolve_image_path(raw_rel, roots)

			car_bbox = _safe_literal_eval(row.get("car_bbox"))
			plate_bbox = _safe_literal_eval(row.get("plate_bbox"))
			char_bbox = _safe_literal_eval(row.get("character_bbox"))

			record = LicensePlateRecord(
				index=idx,
				plate_text=plate_text,
				plate_image_path=plate_path if plate_path else Path(plate_rel),
				raw_image_path=raw_path,
				cameras_plate_no=row.get("cameras_plateNo1"),
				province_code=row.get("province_code"),
				province_description=row.get("province_description"),
				is_validate=is_validate_flag,
				car_bbox=_flatten_single_box(car_bbox),
				plate_bbox=_flatten_single_box(plate_bbox),
				character_bboxes=_normalize_char_boxes(char_bbox),
				metadata={
					"transactionDate": row.get("transactionDate"),
					"brand_description": row.get("brand_description"),
					"colors_code": row.get("colors_code"),
					"colors_description": row.get("colors_description"),
					"vehicleClass": row.get("vehicleClass"),
				},
			)
			records.append(record)

	logger.info("Loaded %d records (require_validate=%s, drop_missing=%s)", len(records), require_validate, drop_missing_images)
	return records


def _flatten_single_box(box: Any) -> Optional[List[float]]:
	"""Convert nested [[x1,y1,x2,y2]] to a flat list or return None."""

	if box is None:
		return None
	if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
		return [float(v) for v in box]
	if isinstance(box, (list, tuple)) and box and isinstance(box[0], (list, tuple)):
		first = box[0]
		if len(first) == 4:
			return [float(v) for v in first]
	return None


def _normalize_char_boxes(char_boxes: Any) -> Optional[List[List[List[float]]]]:
	if not char_boxes:
		return None
	try:
		normalized: List[List[List[float]]] = []
		for box in char_boxes:
			if isinstance(box, (list, tuple)) and len(box) == 4:
				normalized.append([[float(x), float(y)] for x, y in box])
			elif isinstance(box, (list, tuple)):
				normalized.append([[float(pt[0]), float(pt[1])] for pt in box])
		return normalized
	except (TypeError, ValueError):
		logger.warning("Failed to normalize char boxes: %s", char_boxes)
		return None


def stratified_split(
	records: Sequence[LicensePlateRecord],
	*,
	train_ratio: float = 0.7,
	val_ratio: float = 0.20,
	test_ratio: float = 0.10,
	stratify_fn: Optional[Callable[[LicensePlateRecord], Any]] = None,
	seed: int = 42,
) -> Dict[str, List[LicensePlateRecord]]:
	"""Split records into train/val/test subsets using deterministic shuffling."""

	if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3):
		raise ValueError("train/val/test ratios must sum to 1.0")

	stratify_fn = stratify_fn or (lambda rec: (rec.province_code or "UNK", rec.plate_text[:1]))
	buckets: Dict[Any, List[LicensePlateRecord]] = {}
	for rec in records:
		key = stratify_fn(rec)
		buckets.setdefault(key, []).append(rec)

	rng = random.Random(seed)
	splits: Dict[str, List[LicensePlateRecord]] = {"train": [], "val": [], "test": []}

	for key, bucket in buckets.items():
		rng.shuffle(bucket)
		n = len(bucket)
		n_train = math.floor(n * train_ratio)
		n_val = math.floor(n * val_ratio)
		n_test = n - n_train - n_val

		# Adjust rounding leftovers by moving samples from test -> val -> train
		deficit = n - (n_train + n_val + n_test)
		while deficit > 0:
			if n_test > 0:
				n_test += 1
			elif n_val > 0:
				n_val += 1
			else:
				n_train += 1
			deficit -= 1

		splits["train"].extend(bucket[:n_train])
		splits["val"].extend(bucket[n_train : n_train + n_val])
		splits["test"].extend(bucket[n_train + n_val : n_train + n_val + n_test])

		if logger.isEnabledFor(logging.DEBUG):
			logger.debug(
				"Bucket %s: %d -> train %d / val %d / test %d",
				key,
				n,
				n_train,
				n_val,
				n_test,
			)

	for split_name, items in splits.items():
		logger.info("Split '%s': %d samples", split_name, len(items))

	return splits


def export_manifest(
	splits: Dict[str, Sequence[LicensePlateRecord]],
	output_dir: str | Path,
	*,
	include_metadata: bool = False,
	jsonl: bool = True,
) -> Dict[str, Path]:
	"""Write manifest files for each split.

	Returns a mapping `{split: path_to_manifest}` for convenience. Each manifest
	contains one JSON object per line (`jsonl=True`) or a JSON array
	(`jsonl=False`).
	"""

	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	manifests: Dict[str, Path] = {}
	for split_name, records in splits.items():
		path = output_dir / f"{split_name}.{'jsonl' if jsonl else 'json'}"
		if jsonl:
			with path.open("w", encoding="utf-8") as fp:
				for rec in records:
					json.dump(rec.to_example(include_metadata=include_metadata), fp, ensure_ascii=False)
					fp.write("\n")
		else:
			payload = [rec.to_example(include_metadata=include_metadata) for rec in records]
			with path.open("w", encoding="utf-8") as fp:
				json.dump(payload, fp, ensure_ascii=False, indent=2)

		manifests[split_name] = path
		logger.info("Wrote %s manifest to %s", split_name, path)

	return manifests


def to_hf_dataset_dict(
	splits: Dict[str, Sequence[LicensePlateRecord]],
	*,
	include_metadata: bool = False,
):
	"""Create a HuggingFace Dataset dict without touching disk."""

	try:
		from datasets import Dataset  # type: ignore
	except ImportError as exc:  # pragma: no cover - optional dependency
		raise ImportError(
			"datasets library is required for to_hf_dataset_dict; install with `pip install datasets`"
		) from exc

	hf_splits = {}
	for split_name, records in splits.items():
		examples = [rec.to_example(include_metadata=include_metadata) for rec in records]
		hf_splits[split_name] = Dataset.from_list(examples)
	return hf_splits


def load_plate_crop(record: LicensePlateRecord) -> "PILImage":
	"""Open the plate crop as a RGB Pillow image."""

	if Image is None:
		raise RuntimeError("Pillow is required for load_plate_crop but is not installed")
	with Image.open(record.plate_image_path) as img:
		return img.convert("RGB")


def preview_samples(records: Sequence[LicensePlateRecord], limit: int = 5) -> None:
	"""Log a few representative samples for debugging purposes."""

	for rec in records[:limit]:
		logger.info(
			"Sample idx=%s plate='%s' province=%s image=%s",
			rec.index,
			rec.plate_text,
			rec.province_code,
			rec.plate_image_path,
		)


__all__ = [
	"LicensePlateRecord",
	"load_records",
	"stratified_split",
	"export_manifest",
	"to_hf_dataset_dict",
	"load_plate_crop",
	"preview_samples",
]
