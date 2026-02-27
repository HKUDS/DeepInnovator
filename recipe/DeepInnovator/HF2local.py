#!/usr/bin/env python3
# Convert data from opensource format to type format, matching the output of collect_idea_by_type.py as closely as possible

from __future__ import annotations

import argparse
import json
import os
import uuid
from typing import Any

from datasets import Dataset, load_dataset

TYPE_NAME = "DeepInnovator"


def HF2local(
    opensource_rows: list[dict[str, Any]],
    task_desc: str = "refine a research idea",
) -> list[dict[str, Any]]:
    out = []
    for r in opensource_rows:
       
        context_str = r.get("context", "[]")
        try:
            prompt_list = json.loads(context_str)
        except Exception as e:
            raise ValueError(f"Failed to parse context JSON string: {e}")
        
        if not isinstance(prompt_list, list) or len(prompt_list) != 3:
            raise ValueError(f"Invalid context format, should be a list containing 3 messages")
        
        
        assistant_content = prompt_list[1].get("content", "") if len(prompt_list) > 1 else ""
        layer2_content = ""
        
        if "Here are some advanced insights:" in assistant_content:
            
            parts = assistant_content.split("Here are some advanced insights:")
            if len(parts) > 1:
                
                layer2_part = parts[1].split("Here is the first version of the idea:")[0]
                layer2_content = "Here are some advanced insights:" + layer2_part.strip()
        elif "Here are some basic references:" in assistant_content and "Here is the first version" in assistant_content:
            
            layer2_content = ""
        
        
        ground_truth = r.get("target_paper_idea", "")
        if not ground_truth:
            raise ValueError("target_paper_idea cannot be empty")
        
        #extra_info
        
        extra_info = {
            "prompt": prompt_list,
            "context": layer2_content,  # layer2 extracted from context
            "interaction_kwargs": {"name": TYPE_NAME, "task_desc": task_desc},
        }
        
        out.append({
            "prompt": prompt_list,
            "ground_truth": ground_truth,
            "raw_prompt": prompt_list,  
            "extra_info": extra_info,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "data_source": TYPE_NAME,
            "agent_name": TYPE_NAME + "_agent",
            "index": str(uuid.uuid4()),  
        })
    
    return out


def load_hf_dataset(dataset_path: str, split: str = None) -> list[dict[str, Any]]:
    """Load dataset from HuggingFace"""
    if split:
        ds = load_dataset(dataset_path, split=split)
    else:
        ds = load_dataset(dataset_path)
    return [row for row in ds]


def save_parquet(rows: list[dict[str, Any]], filename: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.parquet")
    ds = Dataset.from_list(rows)
    ds.to_parquet(path)
    print(f"[OK] Written {filename}.parquet -> {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser( )
    ap.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset path (e.g., username/dataset_name)",
    )
    ap.add_argument(
        "--train_split",
        default="train",
        help="Training split name (default: train)",
    )
    ap.add_argument(
        "--val_split",
        default="validation",
        help="Validation split name (default: validation)",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output directory (type format)",
    )
    ap.add_argument(
        "--task_desc",
        default="refine a research idea",
        help="Task description (for extra_info.interaction_kwargs.task_desc)",
    )
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Loading dataset from HuggingFace...")
    print(f"  Dataset path: {args.dataset}")
    print(f"  Training split: {args.train_split}")
    print(f"  Validation split: {args.val_split}")
    
    # Load dataset from HuggingFace
    print(f"[INFO] Downloading training set ({args.train_split})...")
    train_opensource = load_hf_dataset(args.dataset, split=args.train_split)
    
    print(f"[INFO] Downloading validation set ({args.val_split})...")
    val_opensource = load_hf_dataset(args.dataset, split=args.val_split)
    
    print(f"[INFO] Training set: {len(train_opensource)} rows, Validation set: {len(val_opensource)} rows")

    if not train_opensource and not val_opensource:
        print("[WARN] No data found, exiting")
        return

    # Convert to type format
    print("[INFO] Converting data format...")
    train_type = HF2local(train_opensource, args.task_desc)
    val_type = HF2local(val_opensource, args.task_desc)

    # Save type format data
    save_parquet(train_type, "rl_train", out_dir)
    save_parquet(val_type, "rl_validation", out_dir)
    
    print(f"\n[DONE] Saved to {out_dir}")

if __name__ == "__main__":
    main()
