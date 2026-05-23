"""Download public Ethiopia measles vaccine coverage source files.

The script intentionally avoids request-only sources. It downloads only files
available through public URLs or public pages.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW_VACCINE_DIR = ROOT / "data" / "raw" / "vaccine"
PROCESSED_DIR = ROOT / "data" / "processed"

WHO_MCV1_URLS = [
    "https://ghoapi.azureedge.net/api/WHS8_110?$filter=SpatialDim%20eq%20%27ETH%27",
    "https://apps.who.int/gho/athena/api/GHO/WHS8_110.csv?filter=COUNTRY:ETH&profile=verbose",
    "http://apps.who.int/gho/athena/api/GHO/WHS8_110.csv?filter=COUNTRY:ETH&profile=verbose",
    "https://apps.who.int/gho/athena/api/GHO/WHS8_110?filter=COUNTRY:ETH&format=csv&profile=verbose",
    "http://apps.who.int/gho/athena/api/GHO/WHS8_110?filter=COUNTRY:ETH&format=csv&profile=verbose",
]
WHO_MCV2_URLS = [
    "https://ghoapi.azureedge.net/api/MCV2?$filter=SpatialDim%20eq%20%27ETH%27",
    "https://apps.who.int/gho/athena/api/GHO/MCV2.csv?filter=COUNTRY:ETH&profile=verbose",
    "http://apps.who.int/gho/athena/api/GHO/MCV2.csv?filter=COUNTRY:ETH&profile=verbose",
    "https://apps.who.int/gho/athena/api/GHO/MCV2?filter=COUNTRY:ETH&format=csv&profile=verbose",
    "http://apps.who.int/gho/athena/api/GHO/MCV2?filter=COUNTRY:ETH&format=csv&profile=verbose",
]
WHO_PROFILE_URL = (
    "https://cdn.who.int/media/docs/default-source/country-profiles/immunization/"
    "2024-country-profiles/immunization-2024-eth.pdf?download=true&sfvrsn=c37780c9_3"
)
DRYAD_DATASET_URL = "https://datadryad.org/dataset/doi:10.5061/dryad.kk2h14b"

APHI_EPI_RESTRICTED_URL = "https://rdmc.aphi.gov.et/items/a5807fcc-abd2-45dd-b5e7-e73e6667a802/full"
DHS_RESTRICTED_URL = "https://microdata.worldbank.org/catalog/3946"


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
    parser.add_argument("--output-dir", type=Path, default=RAW_VACCINE_DIR)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def request_url(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VMOPS-Measles-MVP/1.0 (+public academic data download)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def looks_like_csv(content: bytes) -> bool:
    sample = content[:500].decode("utf-8", errors="ignore").lower()
    return "," in sample and ("year" in sample or "numeric" in sample or "gho" in sample)


def looks_like_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def odata_json_to_csv_bytes(content: bytes) -> bytes:
    payload = json.loads(content.decode("utf-8"))
    rows = payload.get("value")
    if not isinstance(rows, list) or not rows:
        raise ValueError("OData JSON did not contain a non-empty value array.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def download_first_success(
    source_id: str,
    urls: Iterable[str],
    target: Path,
    validator,
    timeout: int,
    retries: int,
) -> DownloadResult:
    errors: list[str] = []
    for url in urls:
        for attempt in range(retries + 1):
            try:
                content = request_url(url, timeout)
                if "ghoapi.azureedge.net" in url:
                    content = odata_json_to_csv_bytes(content)
                if not validator(content):
                    raise ValueError(f"Downloaded content did not match expected format from {url}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                return DownloadResult(source_id, target, "downloaded", url, len(content), "Downloaded successfully.")
            except Exception as exc:  # noqa: BLE001 - we want a compact download log
                errors.append(f"{url} attempt {attempt + 1}: {exc}")
                if attempt < retries:
                    time.sleep(1 + attempt)
    return DownloadResult(source_id, target, "failed", list(urls)[0], 0, " | ".join(errors[-4:]))


def discover_dryad_sav_url(timeout: int) -> tuple[str | None, str]:
    try:
        html = request_url(DRYAD_DATASET_URL, timeout).decode("utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not open Dryad dataset page: {exc}"

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    candidates = [
        href
        for href in hrefs
        if ".sav" in urllib.parse.unquote(href).lower()
        or "file_stream" in href.lower()
        or "downloads" in href.lower()
    ]
    if not candidates:
        return None, "No .sav/file_stream link found on Dryad page."
    first = urllib.parse.urljoin(DRYAD_DATASET_URL, candidates[0])
    return first, "Discovered from Dryad dataset page."


def download_dryad_sav(output_dir: Path, timeout: int, retries: int) -> DownloadResult:
    target = output_dir / "dryad_menz_lalo_vaccination_raw.sav"
    url, note = discover_dryad_sav_url(timeout)
    if not url:
        return DownloadResult("dryad_menz_lalo_vaccination_raw", target, "failed", DRYAD_DATASET_URL, 0, note)
    return download_first_success(
        "dryad_menz_lalo_vaccination_raw",
        [url],
        target,
        lambda content: len(content) > 1000 and not content[:30].lower().startswith(b"<!doctype html"),
        timeout,
        retries,
    )


def write_download_log(results: list[DownloadResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = output_dir / "vaccine_download_log.csv"
    with log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_id", "status", "target_path", "bytes_written", "url", "notes"],
        )
        writer.writeheader()
        for result in results:
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

    unavailable = [
        {
            "source_id": "aphi_epi_2018_2024",
            "source_name": "APHI Routine Immunization Coverage and Vaccine Utilization Dataset (EPI) from 2018-2024",
            "url": APHI_EPI_RESTRICTED_URL,
            "reason": "Request-copy/restricted access; not downloaded because project constraint excludes request-only sources.",
        },
        {
            "source_id": "ethiopia_mini_dhs_2019_microdata",
            "source_name": "2019 Ethiopia Mini DHS microdata",
            "url": DHS_RESTRICTED_URL,
            "reason": "Requires account/access request; not downloaded because project constraint excludes request-only sources.",
        },
    ]
    unavailable_path = output_dir / "vaccine_sources_unavailable_request_only.json"
    unavailable_path.write_text(json.dumps(unavailable, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    results = [
        download_first_success(
            "who_ethiopia_mcv1_coverage",
            WHO_MCV1_URLS,
            out / "who_ethiopia_mcv1_coverage.csv",
            looks_like_csv,
            args.timeout,
            args.retries,
        ),
        download_first_success(
            "who_ethiopia_mcv2_coverage",
            WHO_MCV2_URLS,
            out / "who_ethiopia_mcv2_coverage.csv",
            looks_like_csv,
            args.timeout,
            args.retries,
        ),
        download_first_success(
            "who_ethiopia_immunization_profile_2024",
            [WHO_PROFILE_URL],
            out / "who_ethiopia_immunization_profile_2024.pdf",
            looks_like_pdf,
            args.timeout,
            args.retries,
        ),
        download_dryad_sav(out, args.timeout, args.retries),
    ]
    write_download_log(results, out)
    print(json.dumps([result.__dict__ | {"target_path": str(result.target_path)} for result in results], indent=2))


if __name__ == "__main__":
    main()
