# ascii64_to_parquet_chunked_largest_first.py
# ASCII_64 "~|~" -> Parquet, largest first, chunked, incremental writer
# No pandas. Preserves header row.

import os
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent  


ascii_dir = ROOT / "ASCII_Files"
out_dir   = ROOT / "Main_Parquet_files"

os.makedirs(out_dir, exist_ok=True)
CHUNK_ROWS = 1_000_000
DELIM = "~|~"
ENCODING = "utf-8"


def list_ascii_files_by_size_desc(folder: str):
    items = []
    for f in os.listdir(folder):
        if f.lower().endswith(".txt"):
            p = os.path.join(folder, f)
            try:
                items.append((os.path.getsize(p), f))
            except FileNotFoundError:
                continue
    items.sort(reverse=True)
    return items


def _split_line(line: str):
    return line.rstrip("\n").split(DELIM)


def _rows_to_table(rows, ncols: int, col_names):
    cols = [[] for _ in range(ncols)]

    for r in rows:
        if len(r) < ncols:
            r = r + [""] * (ncols - len(r))
        elif len(r) > ncols:
            r = r[:ncols]

        for j in range(ncols):
            cols[j].append(r[j])

    arrays = [pa.array(col, type=pa.string()) for col in cols]
    return pa.Table.from_arrays(arrays, names=col_names)


ascii_by_size = list_ascii_files_by_size_desc(ascii_dir)

print("ASCII_64 files (largest first):")
for sz, f in ascii_by_size[:10]:
    print(f"  {sz/1024/1024:,.1f} MB  {f}")

for idx, (sz, f) in enumerate(ascii_by_size, 1):
    ascii_path = os.path.join(ascii_dir, f)
    pq_path = os.path.join(out_dir, os.path.splitext(f)[0] + ".parquet")

    if os.path.exists(pq_path) and os.path.getsize(pq_path) > 0:
        print(f"[{idx}/{len(ascii_by_size)}] Skipping (exists): {f}")
        continue

    print(f"[{idx}/{len(ascii_by_size)}] Converting (size {sz/1024/1024:,.1f} MB): {f}")

    writer = None
    total = 0

    try:
        with open(ascii_path, "r", encoding=ENCODING, errors="replace") as fh:

            # ---- READ HEADER ----
            header_line = fh.readline()
            if not header_line:
                print(f"  EMPTY FILE: {f}")
                continue

            col_names = _split_line(header_line)
            ncols = len(col_names)

            rows = []

            while True:
                while len(rows) < CHUNK_ROWS:
                    line = fh.readline()
                    if not line:
                        break
                    rows.append(_split_line(line))

                if not rows:
                    break

                table = _rows_to_table(rows, ncols=ncols, col_names=col_names)

                if writer is None:
                    writer = pq.ParquetWriter(pq_path, table.schema, compression="snappy")

                writer.write_table(table)

                n = table.num_rows
                total += n
                print(f"  wrote {n:,} rows (total {total:,})")

                rows = []

                if not line:
                    break

        if writer is not None:
            writer.close()

        print(f"  DONE: {pq_path}")

    except Exception as e:
        if writer is not None:
            writer.close()
        if os.path.exists(pq_path):
            os.remove(pq_path)
        print(f"  FAILED: {f}")
        print("  Reason:", repr(e))
        print("  Tip: reduce CHUNK_ROWS (e.g., 50_000) and rerun.")
        continue