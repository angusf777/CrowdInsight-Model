#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the import of required dependencies for the Kickstarter project.

This simple utility script verifies that all necessary packages are installed
and can be imported correctly. It tests imports for key dependencies including
torch, numpy, pandas, matplotlib, and SHAP. Use this script to quickly validate
the environment setup before running the main application scripts.

Example usage:
    python test_imports.py

Author: Angus Fung
Date: April 2025
"""

"""Test dependency imports"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

print("All dependencies successfully imported!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"SHAP version: {shap.__version__}") 