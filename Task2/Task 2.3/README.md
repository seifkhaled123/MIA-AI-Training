# Task 2.3 — The Scouting Engine

This project finds players with FIFA 19 profiles similar to M. Salah using the
same nine attributes as Task 2.2.

It includes:

- z-score standardized and raw-data experiments;
- vectorized NumPy implementations of cosine similarity, Euclidean distance,
  Manhattan distance, and Pearson correlation;
- a top-five list for every metric;
- a consensus successor shortlist;
- a from-scratch PCA projection and highlighted plot;
- a generated written comparison in `REPORT.md`.

No similarity metric or PCA implementation from sklearn or scipy is used.

## Run

From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 scouting_engine.py
```

By default, the script reads `../Task 2.2/data/kl.csv`. A different copy of the
FIFA 19 CSV can be supplied as the first argument:

```bash
python3 scouting_engine.py path/to/fifa19.csv
```

Generated CSV, JSON, and plot files are written to `outputs/`.
