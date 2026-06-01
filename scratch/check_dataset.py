import os
import numpy as np
import torch

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

from recbole.config import Config
from recbole.data import create_dataset

base_dir = r"e:\docker-crash-course\RecSys"
config_path = os.path.join(base_dir, "Model_Engine", "ranking_config.yaml")
config = Config(model="DeepFM", config_file_list=[config_path])
dataset = create_dataset(config)

print(f"Total Items: {dataset.item_num}")
print(f"Total Users: {dataset.user_num}")
print(f"Total Interactions: {len(dataset)}")

# Check items with descriptions
item_feat = dataset.item_feat
desc_col = 'Description'
if desc_col in item_feat.interaction:
    valid_desc_count = (item_feat[desc_col].numpy() > 0).sum()
    print(f"Items with valid descriptions: {valid_desc_count}")
else:
    print("Description column not found in item features")
