import requests
import os
from duckduckgo_search import DDGS
queries = ["Bison","Wild Boar","Elephant","Man","Women"]
limit = 920
base_dir = "D:/Animal Detection/Dataset/Train"
os.makedirs(base_dir, exist_ok=True)

with DDGS() as ddgs:
    for query in queries:
        query_dir = os.path.join(base_dir, query.replace(" ", "_"))
        os.makedirs(query_dir, exist_ok=True)

        print(f"\nSearching for images of '{query}'...")
        results = ddgs.images(query, max_results=limit)

        if not results:
            print(f"No images found for '{query}'")
            continue

        for i, img in enumerate(results):
            img_url = img["image"]
            img_ext = img_url.split(".")[-1].split("?")[0]
            img_path = os.path.join(query_dir, f"{query}_{i}.{img_ext}")

            try:
                response = requests.get(img_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded: {img_path}")
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

print("\nAll downloads completed!")
