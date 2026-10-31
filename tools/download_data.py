"""Download the QASPER v0.3 JSON files from the Allen AI dataset bucket.

Example, fetching the dev split into ``data/``:

    python tools/download_data.py --data-dir data --splits dev

Then rebuild the data files with ``tools/prepare_data.py``. The test split is
held out, so it is only fetched when you ask for it with ``--splits test``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qasper_rag.providers.api_client import tls_context

BUCKET = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
ARCHIVES = {
    "train": f"{BUCKET}/qasper-train-dev-v0.3.tgz",
    "dev": f"{BUCKET}/qasper-train-dev-v0.3.tgz",
    "test": f"{BUCKET}/qasper-test-and-evaluator-v0.3.tgz",
}
SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def setup_args() -> argparse.ArgumentParser:
    """Set up command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--splits", nargs="+", choices=sorted(SPLIT_FILES), default=["dev"]
    )
    return parser


def download(url: str, target: Path) -> None:
    """Download ``url`` to ``target`` unless the file is already there."""
    if target.exists():
        return
    logging.info("Downloading %s", url)
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(url, context=tls_context()) as response:
        partial.write_bytes(response.read())
    partial.replace(target)


def extract_split(archive: Path, split: str, data_dir: Path) -> Path:
    """Copy one split's JSON file out of the archive.

    Only that one member is read, so the archive cannot write anywhere else.

    Parameters
    ----------
    archive : Path
        The downloaded ``.tgz`` file.
    split : str
        ``train``, ``dev`` or ``test``.
    data_dir : Path
        The folder to write the JSON file into.

    Returns
    -------
    Path
        The extracted JSON file.
    """
    target = data_dir / SPLIT_FILES[split]
    with tarfile.open(archive) as bundle:
        member = bundle.extractfile(SPLIT_FILES[split])
        if member is None:
            raise ValueError(f"{archive} has no {SPLIT_FILES[split]}")
        target.write_bytes(member.read())
    return target


def main(args: argparse.Namespace) -> None:
    """Download and extract every requested split."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if (data_dir / SPLIT_FILES[split]).exists():
            logging.info("%s is already in %s", split, data_dir)
            continue
        url = ARCHIVES[split]
        archive = data_dir / url.rsplit("/", 1)[-1]
        download(url, archive)
        logging.info("Wrote %s", extract_split(archive, split, data_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(setup_args().parse_args())
