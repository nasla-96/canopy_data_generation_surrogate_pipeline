#!/usr/bin/env python3
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from maize_canopy_pipeline.ml.neural.train_nn_zoo import main

if __name__ == "__main__":
    main()