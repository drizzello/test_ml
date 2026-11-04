"""Scarica training_setA e training_setB della PhysioNet/CinC Challenge 2019.

Fonte: mirror ufficiale PhysioNet su AWS Open Data (physionet-open), licenza ODbL.
Uso: uv run python fetch_data.py
"""
import concurrent.futures as cf
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/training/"
OUT = Path(__file__).parent / "data" / "raw"


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]+\.psv)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(key):
    dest = OUT / key[len(PREFIX):]
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(f"{BUCKET}/{key}", timeout=60).read()
            dest.write_bytes(data)
            return 1
        except Exception:
            if attempt == 4:
                raise


def main():
    keys = list_keys(PREFIX)
    print(f"{len(keys)} file .psv nel bucket")
    with cf.ThreadPoolExecutor(32) as ex:
        n = sum(ex.map(fetch, keys))
    print(f"scaricati {n} nuovi file")
    for s in ["training_setA", "training_setB"]:
        print(s, len(list((OUT / s).glob("*.psv"))))


if __name__ == "__main__":
    sys.exit(main())
