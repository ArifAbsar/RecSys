import os
import numpy as np

if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_
if not hasattr(np, 'unicode'):
    np.unicode = np.str_

from scipy.sparse import dok_matrix
if not hasattr(dok_matrix, '_update'):
    dok_matrix._update = dok_matrix.update

import torch

original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

from recbole.utils import init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer

def run_training(config_file: str):
    """
    Step 4.3: Model Training Execution
    Runs a RecBole training pipeline using the specified YAML config.
    """
    print(f"\n[TRAINING] Starting Stage: {config_file}")
    
    init_seed(42, True)

    config = Config(
        model=None,
        config_file_list=[config_file]
    )

    print("[TRAINING] Loading Dataset...")
    dataset = create_dataset(config)
    print(dataset)

    print("[TRAINING] Preparing Data (Split/Batch)...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    print(f"[TRAINING] Initializing Model: {config['model']}")
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    print(model)

    trainer = get_trainer(config['trainer'], config['model'])(config, model)

    print("[TRAINING] Training started...")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=True)
    
    print("[TRAINING] Evaluating on Test Data...")
    test_result = trainer.evaluate(test_data)
    
    print("\n[SUMMARY] Training Complete!")
    print(f"Best Valid Result: {best_valid_result}")
    print(f"Test Result: {test_result}")

if __name__ == "__main__":
    base_dir = r"e:\docker-crash-course\RecSys"
    
    # 1. Train Retrieval Model (LightGCN)
    run_training(os.path.join(base_dir, "Model_Engine", "retrieval_config.yaml"))
    
    # 2. Train Ranking Model (DeepFM)
    run_training(os.path.join(base_dir, "Model_Engine", "ranking_config.yaml"))
