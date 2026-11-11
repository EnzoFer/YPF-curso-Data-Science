"""Descarga y prepara una muestra reproducible del censo NYC Open Data.

Uso: python download_dataset.py [--rows 100000]
La muestra se descarga mediante Socrata; requiere conexión a internet.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT = DATA_DIR / "nyc_tree_census_sample.csv"
ENDPOINT = "https://data.cityofnewyork.us/resource/uvpi-gqnh.json"
FIELDS = [
    "tree_id", "tree_dbh", "curb_loc", "boroname", "status", "health", "spc_latin",
    "steward", "guards", "sidewalk", "problems", "root_stone", "root_grate",
    "root_other", "trunk_wire", "trnk_light", "trnk_other", "brch_light",
    "brch_shoe", "brch_other",
]


def fetch_page(offset: int, limit: int) -> list[dict]:
    query = urlencode({
        "$select": ",".join(FIELDS),
        "$where": "status='Alive' AND sidewalk IS NOT NULL AND tree_dbh > 0 AND tree_dbh <= 60",
        "$order": "tree_id",
        "$limit": limit,
        "$offset": offset,
    })
    request = Request(f"{ENDPOINT}?{query}", headers={"User-Agent": "YPF-DataScience-course-project/1.0"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--page-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.rows < 100 or args.page_size < 1:
        parser.error("--rows debe ser >= 100 y --page-size debe ser positivo")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reservoir: list[dict] = []
    seen = 0
    offset = 0
    rng = random.Random(args.seed)
    print("Consultando NYC Open Data; se conserva una muestra aleatoria reproducible...")
    while True:
        page = fetch_page(offset, args.page_size)
        if not page:
            break
        for row in page:
            seen += 1
            if len(reservoir) < args.rows:
                reservoir.append(row)
            else:
                position = rng.randrange(seen)
                if position < args.rows:
                    reservoir[position] = row
        offset += len(page)
        print(f"Registros elegibles leídos: {seen:,}", end="\r", flush=True)
        if len(page) < args.page_size:
            break

    if len(reservoir) < 100:
        raise RuntimeError(f"La API devolvió sólo {len(reservoir)} registros; revisar conexión/filtros.")
    # Transformar los objetos de Socrata al formato usado por los notebooks y API.
    with OUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(reservoir)
    print(f"\nMuestra guardada: {OUT} ({len(reservoir):,} de {seen:,}; semilla {args.seed})")
    print("Fuente: NYC Open Data, 2015 Street Tree Census — Tree Data (dataset uvpi-gqnh).")


if __name__ == "__main__":
    main()
