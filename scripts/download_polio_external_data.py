"""Download public aggregate resources for the polio/AFP preparedness module.

Only public, aggregate, legally downloadable resources are used. Sources that
are too large, account-gated, or unavailable are written to the download log as
manual_needed instead of being silently skipped.
"""

from __future__ import annotations

import csv
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
POLIO_VACCINE_DIR = RAW_DIR / "polio_vaccine"
POLIO_SURVEILLANCE_DIR = RAW_DIR / "polio_surveillance"
WASH_DIR = RAW_DIR / "wash"
POPULATION_DIR = RAW_DIR / "population"
LOG_PATH = RAW_DIR / "polio_external_download_log.csv"
TIMEOUT_SECONDS = 25


socket.setdefaulttimeout(TIMEOUT_SECONDS)


@dataclass
class DownloadResult:
    source_id: str
    source_name: str
    url: str
    output_path: str
    status: str
    role: str
    notes: str


def ensure_dirs() -> None:
    for path in [POLIO_VACCINE_DIR, POLIO_SURVEILLANCE_DIR, WASH_DIR, POPULATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def request_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VMOPS-public-health-demo/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def json_to_csv(data: dict[str, Any], output_path: Path) -> int:
    rows = data.get("value", [])
    if not rows:
        pd.DataFrame().to_csv(output_path, index=False)
        return 0
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return len(rows)


def download_gho_indicator(source_id: str, source_name: str, indicator: str, output_path: Path, role: str) -> DownloadResult:
    filter_expr = urllib.parse.quote("SpatialDim eq 'ETH'")
    url = f"https://ghoapi.azureedge.net/api/{indicator}?$filter={filter_expr}"
    try:
        payload = request_url(url)
        data = json.loads(payload.decode("utf-8"))
        rows = json_to_csv(data, output_path)
        status = "downloaded" if rows else "downloaded_empty"
        notes = f"WHO GHO OData indicator {indicator}; Ethiopia rows: {rows}."
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, json.JSONDecodeError, OSError) as exc:
        status = "failed"
        notes = f"Could not download WHO GHO indicator {indicator}: {type(exc).__name__}: {exc}"
    return DownloadResult(source_id, source_name, url, str(output_path), status, role, notes)


def download_file(source_id: str, source_name: str, url: str, output_path: Path, role: str) -> DownloadResult:
    try:
        payload = request_url(url)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        status = "downloaded"
        notes = f"Downloaded {len(payload)} bytes."
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, OSError) as exc:
        status = "failed"
        notes = f"Could not download: {type(exc).__name__}: {exc}"
    return DownloadResult(source_id, source_name, url, str(output_path), status, role, notes)


def manual_source(source_id: str, source_name: str, url: str, output_path: Path, role: str, notes: str) -> DownloadResult:
    return DownloadResult(source_id, source_name, url, str(output_path), "manual_needed", role, notes)


def write_log(results: list[DownloadResult]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DownloadResult.__annotations__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> int:
    ensure_dirs()
    results: list[DownloadResult] = []

    results.append(
        download_gho_indicator(
            "who_gho_eth_pol3",
            "WHO GHO Ethiopia Pol3 immunization coverage",
            "WHS4_544",
            POLIO_VACCINE_DIR / "who_ethiopia_pol3_coverage.csv",
            "national_annual_polio_vaccine_coverage",
        )
    )
    for source_id, source_name, indicator, filename in [
        ("who_gho_eth_ipv1_candidate", "WHO GHO Ethiopia IPV1 candidate endpoint", "IPV1", "who_ethiopia_ipv1_coverage.csv"),
        ("who_gho_eth_ipv2_candidate", "WHO GHO Ethiopia IPV2 candidate endpoint", "IPV2", "who_ethiopia_ipv2_coverage.csv"),
    ]:
        results.append(
            download_gho_indicator(
                source_id,
                source_name,
                indicator,
                POLIO_VACCINE_DIR / filename,
                "national_annual_polio_vaccine_coverage_candidate",
            )
        )

    results.append(
        download_file(
            "owid_polio_screening_testing",
            "Our World in Data polio screening and testing benchmark",
            "https://ourworldindata.org/grapher/polio-screening-and-testing.csv",
            POLIO_SURVEILLANCE_DIR / "owid_polio_screening_testing.csv",
            "external_surveillance_benchmark",
        )
    )

    for source_id, source_name, indicator, filename, role in [
        ("who_gho_eth_open_defecation", "WHO GHO Ethiopia open defecation percent", "WSH_SANITATION_OD", "who_ethiopia_open_defecation.csv", "wash_sanitation_vulnerability"),
        ("who_gho_eth_basic_sanitation", "WHO GHO Ethiopia basic sanitation percent", "WSH_SANITATION_BASIC", "who_ethiopia_basic_sanitation.csv", "wash_sanitation_vulnerability"),
        ("who_gho_eth_basic_water", "WHO GHO Ethiopia basic drinking-water percent", "WSH_WATER_BASIC", "who_ethiopia_basic_water.csv", "wash_water_vulnerability"),
        ("who_gho_eth_safely_managed_water", "WHO GHO Ethiopia safely managed water percent", "WSH_WATER_SAFELY_MANAGED", "who_ethiopia_safely_managed_water.csv", "wash_water_vulnerability"),
    ]:
        results.append(download_gho_indicator(source_id, source_name, indicator, WASH_DIR / filename, role))

    manual_entries = [
        (
            "worldpop_eth_under15_population",
            "WorldPop Ethiopia age-structured population rasters",
            "https://hub.worldpop.org/",
            POPULATION_DIR / "worldpop_ethiopia_under15_manual_needed.txt",
            "population_denominator",
            "WorldPop age rasters are large and require choosing a year/age-sex product before aggregation to woreda.",
        ),
        (
            "healthsites_ethiopia_facilities",
            "Healthsites Ethiopia health facility locations",
            "https://healthsites.io/",
            RAW_DIR / "health_access" / "healthsites_ethiopia_manual_needed.txt",
            "health_access_proxy",
            "Useful for access/facility-density features; direct country export availability should be checked manually.",
        ),
        (
            "iom_hdx_displacement",
            "IOM/HDX displacement datasets",
            "https://data.humdata.org/",
            RAW_DIR / "displacement" / "iom_hdx_displacement_manual_needed.txt",
            "mobility_displacement_proxy",
            "Use only public aggregate Ethiopia displacement products if direct download is available.",
        ),
        (
            "acled_conflict_access",
            "ACLED conflict/access constraints",
            "https://acleddata.com/data/",
            RAW_DIR / "conflict" / "acled_manual_needed.txt",
            "conflict_access_proxy",
            "ACLED usually requires account/API access; list as manual/account-gated rather than scraping.",
        ),
    ]
    for entry in manual_entries:
        result = manual_source(*entry)
        output = Path(result.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.notes + "\nSource: " + result.url + "\n", encoding="utf-8")
        results.append(result)

    write_log(results)
    print(json.dumps({"download_log": str(LOG_PATH), "results": [r.__dict__ for r in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
