import pickle
from pathlib import Path
from typing import Dict
import numpy as np

def load_dataset(name: str, data_root: str) -> Dict[str, np.ndarray]:
    root = Path(data_root)
    with open(root / f"{name}_data.pkl", "rb") as f:
        return pickle.load(f)