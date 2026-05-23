"""Download public FEWS NET Ethiopia IPC Phase 3+ population estimates."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_IPC_DIR = ROOT / "data" / "raw" / "ipc"

FEWSNET_IPC_PHASE3PLUS_URL = (
    "https://fdw.fews.net/api/ipcpopulationsize.csv"
    "?country_code=ET&phase=3%2B&fields=simple"
)
FEWSNET_API_DOCS = "https://help.fews.net/fdw/fews-net-api"


@dataclass
class DownloadResult:
    source_id: str
    target_path: Path
    status: str
    url: str
    bytes_written: int
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RAW_IPC_DIR)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def request_url(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VMOPS-Measles-MVP/1.0 (+public academic data download)",
            "Accept": "text/csv,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def looks_like_ipc_csv(content: bytes) -> bool:
    sample = content[:1000].decode("utf-8", errors="ignore").lower()
    return "country_code" in sample and "phase" in sample and "low_value" in sample


def download_file(source_id: str, url: str, target: Path, timeout: int, retries: int) -> DownloadResult:
    errors: list[str] = []
    for attempt in range(retries + 1):
        try:
            content = request_url(url, timeout)
            if not looks_like_ipc_csv(content):
                raise ValueError("Downloaded content did not look like FEWS NET IPC population CSV.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return DownloadResult(source_id, target, "downloaded", url, len(content), "Downloaded successfully.")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"attempt {attempt + 1}: {exc}")
            if attempt < retries:
                time.sleep(1 + attempt)
    return DownloadResult(source_id, target, "failed", url, 0, " | ".join(errors))


def write_log(result: DownloadResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "ipc_download_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_id", "status", "target_path", "bytes_written", "url", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": result.source_id,
                "status": result.status,
                "target_path": str(result.target_path),
                "bytes_written": result.bytes_written,
                "url": result.url,
                "notes": result.notes,
            }
        )
    docs = {
        "source_id": "fewsnet_ipc_api_docs",
        "source_name": "FEWS NET Data Warehouse API documentation",
        "url": FEWSNET_API_DOCS,
        "notes": "Documents public unauthenticated CSV endpoints including ipcpopulationsize.",
    }
    (output_dir / "ipc_source_reference.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = download_file(
        "fewsnet_ethiopia_ipc_phase3plus_population",
        FEWSNET_IPC_PHASE3PLUS_URL,
        args.output_dir / "fewsnet_ethiopia_ipc_phase3plus_population.csv",
        args.timeout,
        args.retries,
    )
    write_log(result, args.output_dir)
    print(json.dumps(result.__dict__ | {"target_path": str(result.target_path)}, indent=2))


if __name__ == "__main__":
    main()
