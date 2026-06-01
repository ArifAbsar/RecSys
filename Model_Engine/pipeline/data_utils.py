"""
pipeline/data_utils.py
=======================
Data loading and preparation utilities:
  - resolve_paths          : derive all file paths from PipelineConfig
  - load_recbole_model     : set up RecBole Config, Dataset, and load model checkpoint
  - extract_item_metadata  : pull item IDs, descriptions, and id→desc mapping
  - compute_popularity_scores : log-normalised global popularity array
  - build_campaign_flags   : promoted/clearance sets + boolean item array

Extracted verbatim from run_stage2.py (lines 448-608).
No logic has been changed.
"""

import os
import json

import torch
import numpy as np

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model

from .config import PipelineConfig


def resolve_paths(cfg: PipelineConfig, base_dir: str) -> dict:
    """Return a dict of all file paths derived from PipelineConfig."""
    return {
        "base_dir":     base_dir,
        "config_path":  os.path.join(base_dir, "Model_Engine", cfg.config_filename),
        "cp_dir":       os.path.join(base_dir, "Model_Engine", "checkpoints"),
        "mapping_path": os.path.join(base_dir, "output", cfg.mapping_filename),
        "output_path":  os.path.join(base_dir, "output", "final_recommendations.json"),
    }


def load_intelligence_map(mapping_path: str) -> dict:
    """Load rec_config.json (the intelligence / business goals map)."""
    with open(mapping_path, 'r') as fh:
        return json.load(fh)


def load_recbole_model(config_path: str, cp_dir: str, model_name: str):
    """
    Set up RecBole Config + Dataset, then load the latest model checkpoint.
    Returns (config, dataset, model).
    """
    valid_cps = [
        f for f in os.listdir(cp_dir)
        if f.startswith(model_name) and f.endswith('.pth')
    ]
    if not valid_cps:
        raise FileNotFoundError(f"No {model_name} checkpoints found in {cp_dir}")
    model_path = os.path.join(cp_dir, sorted(valid_cps)[-1])

    config  = Config(model=model_name, config_file_list=[config_path])
    dataset = create_dataset(config)
    data_preparation(config, dataset)  # required for RecBole dataset side-effects

    model = get_model(config['model'])(config, dataset).to(config['device'])
    ckpt  = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    return config, dataset, model


def extract_item_metadata(dataset, intelligence_map: dict) -> tuple:
    """
    Extract item IDs, text descriptions, and the id→description mapping.
    Returns (item_ids, descriptions, item_id_to_desc).
    """
    item_limit = dataset.item_num
    item_ids   = [dataset.id2token(dataset.iid_field, i) for i in range(item_limit)]
    desc_col   = intelligence_map.get('content_features', ['Description'])[0]
    item_feat  = dataset.item_feat
    descriptions: list[str] = []

    for i in range(item_limit):
        val = item_feat[desc_col][i]
        if isinstance(val, torch.Tensor):
            val = val.item()
        if desc_col in dataset.field2token_id and val > 0:
            try:    desc_str = dataset.id2token(desc_col, val)
            except: desc_str = "[Invalid]"
        else:
            desc_str = "[No Description]"
        descriptions.append(desc_str)

    item_id_to_desc = {item_ids[i]: descriptions[i] for i in range(item_limit)}
    return item_ids, descriptions, item_id_to_desc


def compute_popularity_scores(dataset, item_limit: int) -> np.ndarray:
    """
    Compute log-normalised global popularity scores for all items.
    Returns a float array of shape (item_limit,) in [0, 1].
    """
    print("[ENGINE] Computing global popularity scores...")
    iid_f      = dataset.iid_field
    pop_counts = dataset.inter_feat[iid_f].numpy()
    unique, counts = np.unique(pop_counts, return_counts=True)

    item_popularity = np.zeros(item_limit)
    for i, count in zip(unique, counts):
        if i < item_limit:
            item_popularity[i] = count

    # Log-transform and normalize to [0, 1]
    popularity_scores = np.log1p(item_popularity)
    if np.max(popularity_scores) > 0:
        popularity_scores /= np.max(popularity_scores)

    return popularity_scores


def build_campaign_flags(
    item_ids: list,
    intelligence_map: dict,
    item_limit: int,
) -> tuple:
    """
    Build promoted/clearance id sets and the is_campaign_item boolean array.
    Returns (promoted_ids, clearance_ids, is_campaign_item).
    """
    promoted_ids  = set(intelligence_map.get('promoted_item_ids', []))
    clearance_ids = set(intelligence_map.get('clearance_item_ids', []))

    # Combined campaign list for exposure cap immunity
    campaign_ids = promoted_ids.union(clearance_ids)

    is_campaign_item = np.zeros(item_limit, dtype=bool)
    for i, iid in enumerate(item_ids):
        if str(iid) in campaign_ids:
            is_campaign_item[i] = True

    return promoted_ids, clearance_ids, is_campaign_item
