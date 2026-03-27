import pickle, json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE.parents[1] / "benchmark" / "data" / "raw" / "EverMind-AI" / "EverMemBench-Static" / "data"

for scale in ("1M", "10M"):
    src = RAW / scale / "unique_reference.pkl"
    data = pickle.loads(src.read_bytes())
    out_dir = BASE / scale
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "corpus.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for i, doc in enumerate(data):
            f.write(json.dumps({"doc_id": i, "text": doc}, ensure_ascii=False) + "\n")
    print(f"{scale}: {len(data)} docs -> {out_file.relative_to(BASE)}  ({out_file.stat().st_size:,} bytes)")
