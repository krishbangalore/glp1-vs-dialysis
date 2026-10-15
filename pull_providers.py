"""
Part 2, step 3 - Download the per-doctor summary file ("Part D Prescribers by Provider")
for nephrologists. One row per doctor per year with model features such as:
  rural/urban code (Prscrbr_RUCA), total patients and claims, patients' average age,
  average risk score (how sick their patients are), dual-eligible (low-income) counts.

All columns are kept; the model step picks the ones it needs.
Output: data/raw/providers_nephrology_<year>.csv  (~9,300 rows per year)
"""

import csv
import os
import time

import requests

DATASETS = {   # CMS "Part D Prescribers - by Provider" dataset IDs (catalog.data.gov)
    2022: "adcdaca6-0c9c-485f-af21-1abc16104d6c",
    2023: "42888afe-3b85-4a61-a4ee-091f00bd62bc",
    2024: "14d8e8a9-7e9b-4370-a044-bf97c46b4b44",
}
API = "https://data.cms.gov/data-api/v1/dataset/{id}/data"
PAGE = 5000

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")


def get_page(dataset_id, offset):
    params = {"filter[Prscrbr_Type]": "Nephrology", "size": PAGE, "offset": offset}
    for attempt in range(6):
        try:
            r = requests.get(API.format(id=dataset_id), params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/6 in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    raise SystemExit("CMS API kept failing - wait a few minutes and re-run (finished years are kept).")


def pull_year(year, dataset_id):
    out = os.path.join(RAW, f"providers_nephrology_{year}.csv")
    if os.path.exists(out):
        print(f"{year}: already downloaded, skipping")
        return
    rows, offset = [], 0
    while True:
        page = get_page(dataset_id, offset)
        rows.extend(page)
        offset += PAGE
        if len(page) < PAGE:
            break
    if not rows:
        raise SystemExit(f"{year}: no rows returned - check the dataset ID")
    fields = list(dict.fromkeys(k for r in rows for k in r))   # union of all keys, in order
    tmp = out + ".partial"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    print(f"{year}: done, {len(rows):,} nephrologists, {len(fields)} columns")


def main():
    os.makedirs(RAW, exist_ok=True)
    for year, dataset_id in DATASETS.items():
        pull_year(year, dataset_id)
    print("\nDone. Provider files are in data/raw/.")


if __name__ == "__main__":
    main()
