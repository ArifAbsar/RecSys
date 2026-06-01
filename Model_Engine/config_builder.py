import os
import yaml
import json

def build_configs(base_dir: str):
    """
    Step 4.2: The Two-Stage Config Builder (Autonomous Version)
    No hardcoded feature names. Everything derived from BML Config.
    """
    config_path = os.path.join(base_dir, "output", "rec_config.json")
    model_choice_path = os.path.join(base_dir, "Model_Engine", "model_choice.json")
    
    with open(config_path, 'r') as f:
        bml_config = json.load(f)
    
    with open(model_choice_path, 'r') as f:
        suggested_model = json.load(f)["suggested_model"]

    user_id = bml_config.get("identity", [])[0]
    item_id = bml_config.get("filters", [])[0]
    
    # Interaction signals (Rating)
    signals = bml_config.get("interaction_signals", {})
    if not signals:
        raise ValueError("[CONFIG] 'interaction_signals' is empty in rec_config.json — cannot determine rating column.")
    if isinstance(next(iter(signals.values())), dict):
        rating_col = max(signals, key=lambda k: signals[k].get("weight", 0))
    else:
        rating_col = max(signals, key=lambda k: signals[k])

    user_features = [col for col in signals.keys() if col != rating_col and "no" not in col.lower() and "id" not in col.lower()]
    
    item_features = bml_config.get("content_features", [])

    print(f"[CONFIG] Building Autonomous Schema:")
    print(f"  • Target:  {rating_col}")
    print(f"  • User Feats: {user_features}")
    print(f"  • Item Feats: {item_features}")

    common_config = {
        "data_path": os.path.join(base_dir, "Model_Engine"),
        "dataset": "rec",
        "checkpoint_dir": os.path.join(base_dir, "Model_Engine", "checkpoints"),
        "epochs": 10,
        "train_batch_size": 2048,
        "learner": "adam",
        "learning_rate": 0.001,
        "eval_step": 1,
        "stopping_step": 2,
        "metrics": ["Recall", "NDCG", "Precision"],
        "topk": [10, 20, 100],
        "valid_metric": "Recall@20",
        "use_gpu": True,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating",
        "TIME_FIELD": "timestamp",
    }

    retrieval_config = common_config.copy()
    retrieval_config.update({
        "model": suggested_model,
        "embedding_size": 64,
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"]
        },
        "train_neg_sample_args": {"uniform": 1},  # replaces deprecated neg_sampling
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full"
        }
    })
    
    retrieval_yaml_path = os.path.join(base_dir, "Model_Engine", "retrieval_config.yaml")
    with open(retrieval_yaml_path, 'w') as f:
        yaml.dump(retrieval_config, f)

    ranking_config = common_config.copy()
    
    load_col = {
        "inter": ["user_id", "item_id", "rating", "timestamp"]
    }
    
    if user_features:
        load_col["user"] = ["user_id"] + user_features
        
    if item_features:
        load_col["item"] = ["item_id"] + item_features

    ranking_config.update({
        "model": "DeepFM",
        "embedding_size": 16,
        "load_col": load_col,
        "LABEL_FIELD": "rating",
        "threshold": {"rating": 0},
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "uni100"
        }
    })
    
    ranking_yaml_path = os.path.join(base_dir, "Model_Engine", "ranking_config.yaml")
    with open(ranking_yaml_path, 'w') as f:
        yaml.dump(ranking_config, f)
    
    print(f"[CONFIG] Success. Created Retrieval ({suggested_model}) and Ranking (DeepFM) YAMLs.")

if __name__ == "__main__":
    base_dir = r"e:\docker-crash-course\RecSys"
    build_configs(base_dir)
