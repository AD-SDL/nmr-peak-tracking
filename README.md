# Peak Tracking and Grouping for NMR

These notebooks explore algorithms for finding peaks in NMR signals then determining which are associated with each other.

## Installation

Install the Python environment using Anaconda:

```bash
conda env create --file environment.yml
```

## Project Layout

We break the peak detection process into several steps:

1. Initial peak detection
1. Linking peaks together into tracks
1. Joining disconnected tracks belonging to same peaks
1. Grouping tracks to those from the same source.

Algorithms for each step are explored in separate folders.
