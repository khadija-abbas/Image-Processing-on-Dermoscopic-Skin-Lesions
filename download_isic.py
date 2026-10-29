import os
import requests
from concurrent.futures import ThreadPoolExecutor

COLLECTION_ID = 482
MAX_IMAGES    = 2000
OUT_DIR       = "Images"
SIZE          = "full"   # or "thumbnail_256" for small, fast files
WORKERS       = 8

os.makedirs(OUT_DIR, exist_ok=True)
session = requests.Session()

def get_image_list():
    url = "https://api.isic-archive.com/api/v2/images/search/"
    params = {"collections": COLLECTION_ID, "limit": 100}
    items = []
    while url and len(items) < MAX_IMAGES:
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        for img in data["results"]:
            items.append((img["isic_id"], img["files"][SIZE]["url"]))
        url = data.get("next")
        params = None  # 'next' already contains the query string
        print(f"Collected {len(items)} image links")
    return items[:MAX_IMAGES]

def download(item):
    isic_id, url = item
    path = os.path.join(OUT_DIR, f"{isic_id}.jpg")
    if os.path.exists(path):
        return "skipped"
    for _ in range(3):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            return "ok"
        except Exception:
            pass
    return f"failed: {isic_id}"

if __name__ == "__main__":
    items = get_image_list()
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(download, items))
    print("Done.", results.count("ok"), "downloaded,",
          results.count("skipped"), "skipped,",
          sum(r.startswith("failed") for r in results), "failed")