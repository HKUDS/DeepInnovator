# Copyright 2025 DeepInnovator team and/or its affiliates
# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from datasets import Dataset
from utils import clean_idea, clean_json

TYPE_NAME = 'DeepInnovator'


SYSTEM_PROMPT = """

  Name: Idea Quality Improver
  Description: Improve a research idea, making refinements to address weaknesses while preserving the core concept.

  # Task Overview
  1. You will be given some critics from users, you should update the idea based on those critics.
  2. Make improvements to address the weaknesses.
  3. Enhance technical_approach with detailed step-by-step workflow if missing or insufficient.
  
  # Critical Principles

  ## Make Improvement
  - Enhance clarity, specificity, and depth rather than changing direction
  - Do NOT change the fundamental nature of the idea

  # Analysis Steps

  - Address weaknesses found by users
  - **CRITICAL: Enhance or add technical_approach**
    * If `technical_approach` is missing or insufficient, add or enhance it with detailed step-by-step workflow
    * Include: data preparation, model/algorithm design, key technical components, implementation details, evaluation methodology
    * Present as a clear, sequential workflow that enables implementation and validation
  - Maintain coherence between different parts of the idea
  - Verify that improvements don't contradict the core concept

  # Output Requirements

  ## Improved Idea Object
  - Output an `improved_idea` object with the same structure as the original
  - All required fields must be present:
    * `current_limitations`: Refined but preserving core problem. 
    * `idea_summary`: Enhanced sentences maintaining core concept. Focus solely on describing this idea itself.
    * `technical_approach`: Enhanced step-by-step description of the methodological workflow. Must include: data preparation steps, model/algorithm design, key technical components, implementation details, evaluation methodology. Present as a clear, sequential workflow that enables implementation and validation. If missing in original, add it. Remember to add details. For example, if you design a RAG system and want to test it, do not use "test on real-world datsets", use "test on HotpotQA".
    * `novelty_statement`: Enhanced novelty claim
    * `confidence`: Confidence level (may improve if clarity increases)

  ## Improvement Summary Object
  - Output an `improvement_summary` object explaining what was improved:
    * `key_improvements`: Array of specific improvements made
    * `improvement_rationale`: Brief explanation of why improvements were made

  # Output Format
  - Output must be valid JSON with no extra fields or commentary, wrapped in <Idea>...</Idea>
  - Output must be in the following format:
    <Idea>
    {{
      "improved_idea": {{
        "current_limitations": "...",
        "idea_summary": "...",
        "technical_approach": "...",
        "novelty_statement": "...",
        "confidence": "...",
        "improvement_summary": {{
          "key_improvements": ["...", "..."],
          "improvement_rationale": "..."
        }}
      }}
    }}
    </Idea>
"""



USER_PROMPT = """
{layer0}
{layer1}
{layer2}
Here is the first version of the idea:
<Idea>
{idea}
</Idea>
"""


# ---------- IO helpers ----------
def save_parquet(ds_split: Dataset, filename: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.parquet")
    ds_split.to_parquet(path)
    print(f"[OK] Wrote {filename}.parquet → {path} ({len(ds_split)} rows)")


def load_all_json_files_from_dir(dir_path: Path) -> dict[str, Any]:
    """Load all JSON files from a directory and return as a dictionary.
    
    Args:
        dir_path: Path to directory containing JSON files.
        
    Returns:
        Dictionary with filename (without .json extension) as key and file content as value.
        Returns empty dict if directory doesn't exist or has no JSON files.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return {}
    
    json_data = {}
    try:
        # Find all JSON files in the directory
        json_files = sorted(dir_path.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Use filename without extension as key
                    key = json_file.stem
                    json_data[key] = data
            except Exception as e:
                print(f"[WARN] Failed to load {json_file}: {e}")
                continue
    except Exception as e:
        print(f"[WARN] Failed to read directory {dir_path}: {e}")
        return {}
    
    return json_data


# Required fields: "prompt", "ground_truth", "extra_info"
# In "extra_info" dict:
# (1) Rquired: "single_turn_prompt", which is the specific problem used to inform the user simulator,
# (2) Optional: "task_desc" (a short task description),
# (3) Optional: other fields for customized reward computation
def collapse_example(example: dict[str, Any]) -> dict[str, Any]:
    if "prompt" not in example:
        raise ValueError("Missing required 'prompt' field.")

    ground_truth = (
        example.get("ground_truth") or example.get("single_turn_completion") or example.get("completion") or ""
    )

    extra_info = {}
    for k, v in example.items():
        if k in ("prompt", "ground_truth", "extra_info"):
            continue
        extra_info.setdefault(k, v)  # keep extra_info values if keys overlap

    # Build system prompt
    # add system prompt as the beginning of the list
    example["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT}] + \
        [{"role": "assistant", "content": example["prompt"]}] + \
        [{"role": "user", "content": 'Please refine this idea. Remember to output the refined idea in the same json format wrapped in <Idea>...</Idea> as the original idea.'}]


    extra_info.setdefault("prompt", example["prompt"])  # save the original prompt
    
    extra_info.setdefault(
        "interaction_kwargs",
        {
            "name": TYPE_NAME,
            "task_desc": extra_info.pop("task_desc", "general ask-for-assistance task"),
        },
    )
    return {
        "prompt": example["prompt"],
        "ground_truth": ground_truth,
        "raw_prompt": example["prompt"],  # save the original prompt
        "extra_info": extra_info,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "data_source": TYPE_NAME,
        "agent_name": TYPE_NAME + "_agent",
        "index": str(uuid.uuid4()),
    }


def train_test_split(
    dataset: Dataset,
    test_size: float,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Perform random train/validation split
    
    Args:
        dataset: Dataset to split
        test_size: Validation set ratio (between 0.0 and 1.0)
        seed: Random seed
        
    Returns:
        (train_dataset, val_dataset): Training set and validation set
    """
    np.random.seed(seed)
    
    # Generate random indices
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    # Calculate split point
    n_val = max(1, int(len(dataset) * test_size))
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    
    # Create training and validation sets
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    
    print(f"\n[INFO] Final split result: train {len(train_dataset)} samples, validation {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", default="./data/arxiv_data", help="HF dataset path or local dir/file."
    )
    ap.add_argument("--task_desc", default="refine a research idea", help="Task description for the dataset.")
    ap.add_argument("--output_dir", default="./data/train", help="Output directory.")
    ap.add_argument(
        "--validation_size", type=float, default=0.1, help="Validation split size (fraction or absolute int)."
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    ap.add_argument("--num_proc", type=int, default=1, help="Parallel workers for map().")
    ap.add_argument("--dataset_type", default="rl", choices=["rl", "sft"], help="Type of dataset (e.g., 'rl', 'sft').")
    ap.add_argument("--test", type=lambda x: x.lower() in ('true', '1', 'yes'), default=True, help="If True, randomly sample 80 training samples and 8 validation samples.")
    ap.add_argument("--layer1", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False, help="If True, load layer1 data from inner_paper_memory.json and inter_paper_group.json.")
    ap.add_argument("--layer0", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False, help="If True, load all JSON files from layer0/paper_memory/ directory.")
    ap.add_argument("--layer2", type=lambda x: x.lower() in ('true', '1', 'yes'), default=False, help="If True, load all JSON files from layer2/ directory.")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Print subpaths under input_dir
    idea_path = Path(args.input_dir)
    Subpath = []
    if idea_path.exists() and idea_path.is_dir():
        subpaths = sorted([item.name for item in idea_path.iterdir() if item.is_dir()])
        for subpath_id in subpaths:
            subpath_full = f"{args.input_dir}/{subpath_id}"
            Subpath.append(Path(subpath_full))
    else:
        print(f"[WARN] Input directory does not exist or is not a directory: {args.input_dir}")
    
    Alldata = []
    for this_path in Subpath:
        target_path = this_path / "target_paper" / "raw_paper" / "paper_idea.json"
        idea_path = this_path / "insights" / "idea_spark.json"
        try:
            # Process layer0 data
            layer0_content = ""
            if args.layer0:
                try:
                    layer0_dir = this_path / "layer0" / "paper_memory"
                    layer0_data = load_all_json_files_from_dir(layer0_dir)
                    if layer0_data:
                        # Clean paper_path field in each file
                        for key in layer0_data:
                            clean_json(layer0_data[key], "paper_path")
                            clean_json(layer0_data[key], "source_papers")
                        layer0_content = "Here are some paper memories:\n" + json.dumps(layer0_data, ensure_ascii=False, indent=2) + "\n"
                except Exception as e:
                    print(f"[WARN] Failed to load layer0 data from {this_path}: {e}")
                    layer0_content = ""
            
            # Process layer1 data
            layer1_content = ""
            if args.layer1:
                try:
                    inner_paper_memory_path = this_path / "layer1" / "inner_paper_memory.json"
                    inter_paper_group_path = this_path / "layer1" / "inter_paper_group.json"
                    
                    layer1_data = {}
                    if inner_paper_memory_path.exists():
                        with open(inner_paper_memory_path, "r", encoding="utf-8") as f:
                            inner_data = json.load(f)
                            clean_json(inner_data, "paper_path")
                            layer1_data["inner_paper_memory"] = inner_data
                    
                    if inter_paper_group_path.exists():
                        with open(inter_paper_group_path, "r", encoding="utf-8") as f:
                            inter_data = json.load(f)
                            clean_json(inter_data, "paper_path")
                            layer1_data["inter_paper_group"] = inter_data
                    
                    if layer1_data:
                        layer1_content = "Here are some basic references:\n" + json.dumps(layer1_data, ensure_ascii=False, indent=2) + "\n"
                except Exception as e:
                    print(f"[WARN] Failed to load layer1 data from {this_path}: {e}")
                    layer1_content = ""
            
            # Process layer2 data
            layer2_content = ""
            if args.layer2:
                try:
                    layer2_dir = this_path / "layer2"
                    layer2_data = load_all_json_files_from_dir(layer2_dir)
                    if layer2_data:
                        # Clean paper_path field in each file
                        for key in layer2_data:
                            clean_json(layer2_data[key], "paper_path")
                        layer2_content = "Here are some advanced insights:\n" + json.dumps(layer2_data, ensure_ascii=False, indent=2) + "\n"
                except Exception as e:
                    print(f"[WARN] Failed to load layer2 data from {this_path}: {e}")
                    layer2_content = ""
            
            with open(target_path, "r") as f:
                target_idea = json.load(f).get("ideas", [])[0]
                target_idea, _ = clean_idea(target_idea)
                target_idea = json.dumps(target_idea, ensure_ascii=False)
            
            with open(idea_path, "r") as f:
                input_ideas = json.load(f).get("ideas", [])
            for this_idea in input_ideas:
                this_idea, _ = clean_idea(this_idea)
                this_data = {
                    "prompt": USER_PROMPT.format(
                        layer0=layer0_content,
                        layer1=layer1_content,
                        layer2=layer2_content,
                        idea=json.dumps(this_idea, ensure_ascii=False)
                    ),
                    "ground_truth": target_idea,
                }
                Alldata.append(this_data)
            print(f"[INFO] Processed {this_path} successfully")
        except Exception as e:
            continue

    ds_all = Dataset.from_list(Alldata)

    if args.dataset_type == "rl":
        # RL (reinforcement learning) dataset processing pipeline
        
        # If multiple splits exist, merge them before conversion/splitting
        # Add task description field to each example
        ds_all = ds_all.map(lambda x: {"task_desc": args.task_desc}, num_proc=args.num_proc)

        # Convert examples to standardized format
        print(f"[INFO] Collapsing to formatted fields on {len(ds_all)} rows…")
        ds_all = ds_all.map(
            function=collapse_example,  # Conversion function
            remove_columns=ds_all.column_names,  # Remove original columns (collapse_example creates new columns)
            num_proc=args.num_proc,  # Number of parallel processing processes
        )

        # Define deduplication function: deduplicate based on prompt content
        def dedup_by_prompt(dataset):
            """
            Deduplicate based on prompt content
            
            Serialize prompt as JSON string as unique key, remove duplicate examples.
            This ensures each unique conversation prompt retains only one example.
            
            Args:
                dataset: Dataset to deduplicate
            
            Returns:
                Dataset: Deduplicated dataset
            """
            seen = set()  # Store seen prompt keys
            unique_rows = []  # Store unique examples
            
            for ex in dataset:
                # Serialize prompt as JSON string as unique key
                # sort_keys=True ensures prompts with same content generate same key
                prompt_key = json.dumps(ex["prompt"], sort_keys=True, ensure_ascii=False)
                
                # If this prompt hasn't been seen, add to result
                if prompt_key not in seen:
                    seen.add(prompt_key)
                    unique_rows.append(ex)
            
            # Create new dataset from unique rows list
            return Dataset.from_list(unique_rows)

        # Execute deduplication
        ds_all = dedup_by_prompt(ds_all)

    elif args.dataset_type == "sft":
        raise NotImplementedError("SFT is not implemented for DeepInnovator.")

    # Use random sampling for dataset splitting
    print(f"[INFO] Splitting with validation_size={args.validation_size}, seed={args.seed}")
    
    if args.test:
        # Test mode: first perform random split, then sample fixed number
        print(f"[INFO] Test mode enabled: first perform random split, then sample fixed number")
        train_ds, val_ds = train_test_split(
            ds_all, 
            test_size=args.validation_size, 
            seed=args.seed
        )
        
        # Sample fixed number from each set
        def sample_random(dataset, target_size, seed):
            """Sample specified number of samples from dataset randomly"""
            if len(dataset) == 0:
                return dataset.select([])
            
            np.random.seed(seed)
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)
            sampled_indices = indices[:min(target_size, len(dataset))].tolist()
            return dataset.select(sampled_indices)
        
        # Sample with fixed numbers
        train_ds = sample_random(train_ds, min(80, len(train_ds)), args.seed)
        val_ds = sample_random(val_ds, min(8, len(val_ds)), args.seed)
        print(f"[INFO] Test mode: after sampling, train {len(train_ds)} samples, validation {len(val_ds)} samples")
    else:
        # Normal mode: directly use random split
        train_ds, val_ds = train_test_split(
            ds_all, 
            test_size=args.validation_size, 
            seed=args.seed
        )
    
    print(train_ds, val_ds)

    # Print an example data
    # if len(train_ds) > 0:
    #     print("\n" + "="*80)
    #     print("[INFO] Example data (first sample in training set):")
    #     print("="*80)
    #     example = train_ds[0]
    #     print(json.dumps(example, indent=2, ensure_ascii=False))
    #     print("="*80 + "\n")

    save_parquet(train_ds, f"{args.dataset_type}_train", out_dir)
    save_parquet(val_ds, f"{args.dataset_type}_validation", out_dir)
    print(f"[DONE] {args.dataset_type}_train.parquet and {args.dataset_type}_validation.parquet written.")


if __name__ == "__main__":
    main()
