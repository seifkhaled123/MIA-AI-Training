# FIFA 19 player performance PCA

This project implements Principal Component Analysis from scratch on the
FIFA 19 Complete Player Dataset. It includes:

- a NumPy implementation without `sklearn`;
- a pure-Python list-of-lists implementation of standardization, transpose,
  matrix multiplication, covariance, and projection;
- validation and timing of both pipelines;
- a PC1/PC2 player scatter plot colored by `Position`.

## Run

From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 pca_analysis.py
```

Alternatively, run all cells in `test.ipynb`. Generated files are placed in
`outputs/`, and the written analysis is in `REPORT.md`.
