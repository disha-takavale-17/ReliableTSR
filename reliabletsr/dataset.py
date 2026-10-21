"""Download only official PubTables structure test archives; retain selected pairs."""
import json
import random
import shutil
import tarfile
from pathlib import Path

from . import DATASET_ID, DATASET_REVISION

ANNOTATIONS = "PubTables-1M-Structure_Annotations_Test.tar.gz"
IMAGES = "PubTables-1M-Structure_Images_Test.tar.gz"
ESTIMATED_BYTES = {ANNOTATIONS: 29_900_000, IMAGES: 2_690_000_000}


def archive_url(name):
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}"


def download(name, root, min_free_gb=6):
    import requests
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / name
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    free = shutil.disk_usage(root).free
    required = max(ESTIMATED_BYTES[name] - offset + 1_000_000_000, int(min_free_gb * 1e9))
    if free < required:
        raise RuntimeError(f"Insufficient disk space: {free/1e9:.2f} GB free; need at least {required/1e9:.2f} GB")
    url = archive_url(name)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        resume = offset > 0 and response.status_code == 206
        reported = int(response.headers.get("Content-Length", "0"))
        print(f"Downloading {name}: official listing ~{ESTIMATED_BYTES[name]/1e9:.2f} GB; "
              f"HTTP size {reported/1e9:.2f} GB; free {free/1e9:.2f} GB")
        with partial.open("ab" if resume else "wb") as out:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    out.write(chunk)
        partial.replace(destination)
    if not tarfile.is_tarfile(destination):
        raise RuntimeError(f"Downloaded file is not a tar archive: {destination}")
    return destination


def _safe_member(member):
    return member.isfile() and not member.issym() and not member.islnk() and ".." not in Path(member.name).parts


def prepare(root="data", count=400, dev_count=250, seed=1701, keep_archives=False):
    if not (300 <= count <= 500 and 0 < dev_count < count):
        raise ValueError("Use 300–500 pairs and a nonempty development/final split")
    root = Path(root)
    raw = root / "raw"
    pairs = root / "pairs"
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (existing.get("count"), existing.get("development_count"), existing.get("seed")) == (count, dev_count, seed):
            complete = all((pairs / item["group"] / "xml" / (item["stem"] + ".xml")).exists() and
                           any((pairs / item["group"] / "images").glob(item["stem"] + ".*"))
                           for item in existing["pairs"])
            if complete:
                return existing
    annotation_archive = download(ANNOTATIONS, raw)
    with tarfile.open(annotation_archive, "r:gz") as archive:
        names = sorted({Path(m.name).stem for m in archive if _safe_member(m) and m.name.lower().endswith(".xml")})
    if len(names) < count:
        raise RuntimeError(f"Only {len(names)} test XML files found")
    selected = random.Random(seed).sample(names, count)
    groups = {name: ("development" if i < dev_count else "final") for i, name in enumerate(selected)}
    selected_set = set(selected)
    pairs.mkdir(parents=True, exist_ok=True)
    extracted_xml = set()
    with tarfile.open(annotation_archive, "r:gz") as archive:
        for member in archive:
            stem = Path(member.name).stem
            if _safe_member(member) and stem in selected_set and member.name.lower().endswith(".xml"):
                target = pairs / groups[stem] / "xml" / (stem + ".xml")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_xml.add(stem)
    if extracted_xml != selected_set:
        raise RuntimeError(f"Missing {len(selected_set - extracted_xml)} selected XML files")
    image_archive = download(IMAGES, raw)
    extracted_images = set()
    with tarfile.open(image_archive, "r:gz") as archive:
        for member in archive:
            stem = Path(member.name).stem
            if _safe_member(member) and stem in selected_set and member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                suffix = Path(member.name).suffix.lower()
                target = pairs / groups[stem] / "images" / (stem + suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_images.add(stem)
    if extracted_images != selected_set:
        raise RuntimeError(f"Missing {len(selected_set - extracted_images)} selected images")
    manifest = {"dataset": DATASET_ID, "revision": DATASET_REVISION, "seed": seed,
                "count": count, "development_count": dev_count, "final_count": count - dev_count,
                "archives": {name: {"url": archive_url(name), "bytes": (raw/name).stat().st_size}
                             for name in (ANNOTATIONS, IMAGES)},
                "pairs": [{"stem": name, "group": groups[name]} for name in selected]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not keep_archives:
        annotation_archive.unlink()
        image_archive.unlink()
    return manifest
