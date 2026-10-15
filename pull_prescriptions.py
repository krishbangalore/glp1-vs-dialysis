"""
Part 2, step 1 - Download every Medicare Part D prescription line written by
nephrologists, 2019-2024, from the CMS "Part D Prescribers by Provider and Drug" API.

We keep ALL drugs (not just GLP-1s) because what else a doctor prescribes
(insulin, metformin, SGLT2s...) becomes a feature for the model later.

Note: CMS hides any doctor-drug pair with fewer than 11 claims, so "prescribes X"
in this data really means "wrote 11+ Medicare prescriptions for X that year".

Output: data/raw/partd_nephrology_<year>.csv  (~265k rows per year)
Re-running skips years already downloaded; a crashed year restarts cleanly.
"""

import csv
import os
import time

import requests

DATASETS = {   # CMS dataset IDs per data year (from catalog.data.gov)
    2019: "2a6705e6-7a1e-460c-ba22-35249a531918",
    2020: "7795fe20-e80e-435a-a9ed-d2d65e05feeb",
    2021: "f68114ed-f854-4ffc-9c6e-ed78b5e2f8d0",
    2022: "b101b457-ffa4-49bb-8fd9-27c1266086e2",
    2023: "e54db557-cd82-4e91-a0fe-61aad5865d69",
    2024: "9552739e-3d05-4c1b-8eff-ecabf391e2e5",
}
API = "https://data.cms.gov/data-api/v1/dataset/{id}/data"
SPECIALTY = "Nephrology"
PAGE = 5000
KEEP = ["Prscrbr_NPI", "Prscrbr_City", "Prscrbr_State_Abrvtn", "Prscrbr_Type",
        "Brnd_Name", "Gnrc_Name", "Tot_Clms", "Tot_30day_Fills", "Tot_Day_Suply",
        "Tot_Drug_Cst", "Tot_Benes"]

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")


def get_page(dataset_id, offset):
    params = {"filter[Prscrbr_Type]": SPECIALTY, "size": PAGE, "offset": offset}
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
    out = os.path.join(RAW, f"partd_nephrology_{year}.csv")
    if os.path.exists(out):
        print(f"{year}: already downloaded, skipping")
        return
    tmp = out + ".partial"
    n, offset = 0, 0
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP, extrasaction="ignore")
        w.writeheader()
        while True:
            rows = get_page(dataset_id, offset)
            if not rows:
                break
            w.writerows(rows)
            n += len(rows)
            offset += PAGE
            print(f"  {year}: {n:,} rows", end="\r")
            if len(rows) < PAGE:
                break
    os.replace(tmp, out)              # only a complete year gets the final filename
    print(f"{year}: done, {n:,} rows            ")


def main():
    os.makedirs(RAW, exist_ok=True)
    for year, dataset_id in DATASETS.items():
        pull_year(year, dataset_id)
    print("\nAll years downloaded to data/raw/. Next: python adoption_summary.py")


if __name__ == "__main__":
    main()
