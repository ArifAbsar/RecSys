import argparse
from domain_loader import build_target_registry
from text_builder.text_builder import build_target_text

def main():
    parser = argparse.ArgumentParser(description="Run Domain Loader and Text Builder together.")
    parser.add_argument("--path", required=True, help="Path to the schema YAML file (e.g. ecommerce/schema.yaml)")
    args = parser.parse_args()

    print(f"--- Step 1: Loading Domain Registry from {args.path} ---")
    targets = build_target_registry(args.path)
    print(f"Loaded {len(targets)} target fields.\n")
    print(f"--- Step 2: Building Text for Embeddings ---")
    for i, target in enumerate(targets):
        text = build_target_text(target)
        print(f"FIELD {i+1}: {target.get('field_id')}")
        print("-" * 20)
        print(text)
        print("-" * 20 + "\n")
        
        if i >= 4: 
            print("... (truncated for brevity) ...")
            break

if __name__ == "__main__":
    main()
