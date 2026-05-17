"""
scrape.py - Flatfox public-listing API scraper.

Reads the JSON feed at flatfox.ch/api/v1/public-listing/, filters to
APARTMENT-only rentals in Switzerland, derives canton from PLZ via
pgeocode, and writes data/raw.csv (append-safe, deduped by listing id).

Originally targeted homegate.ch, which sits behind a DataDome CAPTCHA
and returns 403 to headless browsers. Pivoted to Flatfox, which exposes
a public JSON API. No browser needed; stdlib urllib only.

About 5% of all Flatfox listings under offer_type=RENT are residential
apartments; the rest are parking, commercial, etc. The APARTMENT filter
runs client-side because the API ignores the server-side category param.

price_chf is reconstructed gross rent: rent_gross if set, otherwise
rent_net + rent_charges, otherwise the row is dropped. Capped at
MAX_MONTHLY_RENT_CHF to reject sale prices that occasionally leak into
the rental feed (e.g. CHF 980,000 as price_display on a RENT listing).

latitude and longitude come from Flatfox directly (100% populated, real
address coordinates). Canton is derived from PLZ via pgeocode.

Rows are buffered in memory and written once at the end so concurrent
processes (Excel, AV scanners) can't lock raw.csv mid-write.

Usage:
    python scrape.py                                            # all 26 cantons, up to 5000 listings
    python scrape.py --max-listings 100                         # smoke test
    python scrape.py --cantons Bern,Zurich,Vaud,Geneva,Aargau,Lucerne   # filter to spec's 6 cantons
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pgeocode


API_URL = "https://flatfox.ch/api/v1/public-listing/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)
HOST = "https://flatfox.ch"

DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "raw.csv"

FIELDS = [
    "id", "title", "price_chf", "rooms", "size_m2", "plz",
    "city", "canton", "street", "listing_type",
    "latitude", "longitude",          # taken directly from Flatfox (100% populated, exact-address centroid)
    "url", "scraped_at",
]

# Hard upper bound on monthly rent. Flatfox's RENT feed occasionally has sale
# prices (e.g. CHF 980,000) leak into price_display; this filter rejects them.
MAX_MONTHLY_RENT_CHF = 20_000

# All 26 Swiss cantons (ISO state_code from pgeocode → human-readable label).
CANTON_CODE_TO_LABEL: dict[str, str] = {
    "AG": "Aargau", "AI": "Appenzell Innerrhoden", "AR": "Appenzell Ausserrhoden",
    "BE": "Bern", "BL": "Basel-Landschaft", "BS": "Basel-Stadt",
    "FR": "Fribourg", "GE": "Geneva", "GL": "Glarus", "GR": "Graubünden",
    "JU": "Jura", "LU": "Lucerne", "NE": "Neuchâtel", "NW": "Nidwalden",
    "OW": "Obwalden", "SG": "St. Gallen", "SH": "Schaffhausen", "SO": "Solothurn",
    "SZ": "Schwyz", "TG": "Thurgau", "TI": "Ticino", "UR": "Uri",
    "VD": "Vaud", "VS": "Valais", "ZG": "Zug", "ZH": "Zurich",
}

# Original spec's 6 cantons of interest - kept as an opt-in filter via --cantons.
DEFAULT_CANTONS_OF_INTEREST = {"Bern", "Zurich", "Vaud", "Geneva", "Aargau", "Lucerne"}

# Apartment-only. Houses and secondary residences are different markets with
# different price dynamics; including them would weaken the per-cluster regression.
RESIDENTIAL_CATEGORIES = {"APARTMENT"}


# ---------- HTTP ----------

def fetch_page(offset: int, limit: int, timeout: int = 30) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "ordering": "-last_published_at",
        "offer_type": "RENT",
        "object_category": "APARTMENT",  # ignored server-side, but harmless
        "limit": str(limit),
        "offset": str(offset),
    })
    url = f"{API_URL}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Language": "de-CH,de;q=0.9,en;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------- Parsing ----------

def _to_int(x: Any) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class CantonLookup:
    """PLZ → ISO canton code, cached. Wraps pgeocode.Nominatim('ch')."""

    def __init__(self) -> None:
        self.nom = pgeocode.Nominatim("ch")
        self.cache: dict[str, str | None] = {}

    def state_code(self, plz: str) -> str | None:
        if not plz:
            return None
        plz_str = str(plz).strip().zfill(4)
        if plz_str in self.cache:
            return self.cache[plz_str]
        try:
            row = self.nom.query_postal_code(plz_str)
            code = row.get("state_code")
            # pandas Series scalar → str, NaN-safe
            if code is None or (isinstance(code, float) and code != code):
                self.cache[plz_str] = None
                return None
            result = str(code)
            self.cache[plz_str] = result
            return result
        except Exception:
            self.cache[plz_str] = None
            return None


def parse_listing(item: dict[str, Any], canton_lookup: CantonLookup) -> dict[str, Any] | None:
    """Flatten one Flatfox listing into our row schema. Returns None to drop."""
    pk = item.get("pk")
    if not pk:
        return None

    # Filter: residential rentals only, monthly rent, Switzerland.
    if item.get("object_category") not in RESIDENTIAL_CATEGORIES:
        return None
    if item.get("price_unit") and item["price_unit"] != "monthly":
        return None
    if item.get("country") and item["country"] != "CH":
        return None

    # Reconstructed gross rent (Bruttomiete) - gives one consistent semantic
    # across all kept rows. Drop if no gross figure can be reconstructed.
    gross = item.get("rent_gross")
    if gross is None:
        net = item.get("rent_net")
        charges = item.get("rent_charges")
        if net is not None and charges is not None:
            gross = net + charges
        else:
            return None
    price_chf = _to_int(gross)
    if price_chf is None or price_chf > MAX_MONTHLY_RENT_CHF:
        return None

    rooms = _to_float(item.get("number_of_rooms"))
    size = _to_float(item.get("surface_living"))
    lat = _to_float(item.get("latitude"))
    lon = _to_float(item.get("longitude"))

    plz_raw = item.get("zipcode")
    plz = str(plz_raw).strip().zfill(4) if plz_raw else ""
    city = (item.get("city") or "").strip()
    street = (item.get("street") or "").strip()

    state_code = canton_lookup.state_code(plz) if plz else None
    canton_label = CANTON_CODE_TO_LABEL.get(state_code or "", state_code or "")

    title = (item.get("public_title") or item.get("short_title") or "").strip()
    listing_type = str(item.get("object_type") or "APARTMENT")

    rel = item.get("url") or f"/{pk}/"
    url = f"{HOST}{rel}"

    return {
        "id": str(pk),
        "title": title,
        "price_chf": price_chf,
        "rooms": rooms,
        "size_m2": size,
        "plz": plz,
        "city": city,
        "canton": canton_label,
        "street": street,
        "listing_type": listing_type,
        "latitude": lat,
        "longitude": lon,
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------- CSV I/O ----------

def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("id"):
                seen.add(row["id"])
    return seen


def write_all(path: Path, rows: list[dict[str, Any]], append_mode: bool) -> None:
    """Write the buffer to CSV in one shot. If append_mode and file exists,
    append without a fresh header; otherwise overwrite with header."""
    if not rows:
        return
    write_header = not (append_mode and path.exists())
    mode = "a" if (append_mode and path.exists()) else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------- Main loop ----------

def crawl(max_listings: int, page_size: int, delay: float, canton_filter: set[str] | None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen = load_existing_ids(RAW_CSV)
    canton_lookup = CantonLookup()

    buffered: list[dict[str, Any]] = []
    offset = 0
    page = 0
    total_fetched = 0
    drops = {"non_residential": 0, "wrong_canton": 0, "duplicate": 0, "bad_data": 0}

    while len(buffered) < max_listings:
        try:
            data = fetch_page(offset, page_size)
        except urllib.error.HTTPError as e:
            print(f"[ERROR] HTTP {e.code} at offset {offset}: {e}", file=sys.stderr)
            break
        except Exception as e:
            print(f"[ERROR] fetch failed at offset {offset}: {e}", file=sys.stderr)
            break

        results = data.get("results") or []
        total_fetched += len(results)
        if not results:
            print(f"[page {page:3}] empty results - done.")
            break

        kept_this_page = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            # Categorise pre-parse so we can count drop reasons cleanly.
            if item.get("object_category") not in RESIDENTIAL_CATEGORIES:
                drops["non_residential"] += 1
                continue
            row = parse_listing(item, canton_lookup)
            if row is None:
                drops["bad_data"] += 1
                continue
            if canton_filter is not None and row["canton"] not in canton_filter:
                drops["wrong_canton"] += 1
                continue
            if row["id"] in seen:
                drops["duplicate"] += 1
                continue
            seen.add(row["id"])
            buffered.append(row)
            kept_this_page += 1

        page += 1
        print(
            f"[page {page:3} offset={offset:5}] fetched {len(results):3}, "
            f"kept {kept_this_page:3}, buffered={len(buffered)}"
        )

        if not data.get("next"):
            print(f"[page {page:3}] no next page - done.")
            break
        offset += page_size
        if len(buffered) < max_listings:
            time.sleep(delay)

    print(f"\nWriting {len(buffered)} new rows to {RAW_CSV} ...")
    write_all(RAW_CSV, buffered, append_mode=True)
    print(f"Done.")
    print(f"  pages crawled       : {page}")
    print(f"  raw items fetched   : {total_fetched}")
    print(f"  new rows saved      : {len(buffered)}")
    print(f"  drops - non-residential : {drops['non_residential']}")
    print(f"  drops - wrong canton    : {drops['wrong_canton']}")
    print(f"  drops - duplicate id    : {drops['duplicate']}")
    print(f"  drops - bad data        : {drops['bad_data']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-listings", type=int, default=5000,
                        help="stop after this many new listings (default 5000; set to a large number to crawl to exhaustion)")
    parser.add_argument("--page-size", type=int, default=50,
                        help="API page size, max 50 (default 50)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between API calls (default 1.0)")
    parser.add_argument("--cantons", default=None,
                        help="comma-separated canton names to KEEP, e.g. 'Bern,Zurich,Vaud,Geneva,Aargau,Lucerne'. "
                             "Default: no filter (all 26 cantons).")
    args = parser.parse_args()

    canton_filter: set[str] | None = None
    if args.cantons:
        canton_filter = {c.strip() for c in args.cantons.split(",") if c.strip()}

    crawl(
        max_listings=args.max_listings,
        page_size=min(args.page_size, 50),
        delay=args.delay,
        canton_filter=canton_filter,
    )


if __name__ == "__main__":
    main()
