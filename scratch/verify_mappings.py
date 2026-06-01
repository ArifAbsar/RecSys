import os
import sys
import numpy as np

# Fix for NumPy 2.0 compatibility in RecBole
for old_name, new_type in [
    ('float_', np.float64),
    ('int_', np.int64),
    ('bool_', np.bool_),
    ('complex_', np.complex128),
    ('unicode_', np.str_),
]:
    if not hasattr(np, old_name):
        setattr(np, old_name, new_type)

from scipy.sparse import dok_matrix
if not hasattr(dok_matrix, '_update'):
    dok_matrix._update = dok_matrix.update

# Ensure Model_Engine path is in sys.path
sys.path.append(r"e:\docker-crash-course\RecSys\Model_Engine")

from pipeline.data_utils import load_recbole_model
from pipeline.config import PipelineConfig

def verify():
    cfg = PipelineConfig()
    base_dir = r"e:\docker-crash-course\RecSys"
    
    rank_config_path = os.path.join(base_dir, "Model_Engine", cfg.config_filename)
    cp_dir = os.path.join(base_dir, "Model_Engine", "checkpoints")
    
    print("Loading Ranking (DeepFM) model and dataset...")
    rank_config, rank_dataset, rank_model = load_recbole_model(rank_config_path, cp_dir, cfg.base_model_name)
    
    print("\n--- MODEL FIELDS ---")
    print(f"Token field names: {getattr(rank_model, 'token_field_names', 'N/A')}")
    print(f"Float field names: {getattr(rank_model, 'float_field_names', 'N/A')}")
    print(f"Token field dims: {getattr(rank_model, 'token_field_dims', 'N/A')}")
    print(f"Float field dims: {getattr(rank_model, 'float_field_dims', 'N/A')}")
    
    print("\n--- DATASET FIELDS ---")
    for field, ftype in rank_dataset.field2type.items():
        print(f"Field: {field}, Type: {ftype}")

if __name__ == "__main__":
    verify()
