#!/usr/bin/env python3
"""
"""

# Standard library imports for file handling, system operations, and data structures
import argparse
import json
import logging
import os
import sys
from collections import defaultdict          # For grouping papers and aggregating updates
from datetime import datetime               # For timestamp management and key dates
from itertools import count as count_iterator  # For generating unique sequential IDs
from pathlib import Path                    # For cross-platform file path handling

# Third-party library imports for configuration management and API communication
from dotenv import load_dotenv              # For loading environment variables from .env files
from omegaconf import OmegaConf             # For hierarchical configuration management with YAML support
from openai import OpenAI                   # For OpenAI-compatible API communication (supports various providers)

# Add parent directory to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file (API keys, base URLs, etc.)
load_dotenv()

# Project-specific imports
from data_prepare.utils import (
    load_config,
    call_agent,
    Paper_Profile,
    Paper,
    extract_json_from_markdown,
    process_stream_output,
    setup_logger
)
        


def get_target_paper_idea(
    target_paper_md: str, 
    output_path: str):
    config = load_config()
    agent_config = config['agents']['paper_idea_extractor']
    agent_prompt = agent_config['prompt'].format(
        paper=target_paper_md,
        schema=json.dumps(
            OmegaConf.to_container(agent_config['schema'], resolve=True),
            indent=2,
            ensure_ascii=False
        )
    )
    if os.path.exists(output_path):
        return
    result = call_agent('paper_idea_extractor', agent_prompt, config)
    # print(result)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("save to ", output_path)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ideas from target papers")
    parser.add_argument(
        "--datapath", "-d",
        type=str,
        default="data/arxiv_data",
        help="Path to arxiv data directory (default: data/arxiv_data)",
    )
    args = parser.parse_args()
    DATAPATH = args.datapath
    subdirs = [name for name in os.listdir(DATAPATH) if os.path.isdir(os.path.join(DATAPATH, name))]
    idx = 0
    for dir in subdirs:
        print(f"total {len(subdirs)}, idx {idx}")
        idx += 1
        try:
            paper_md = Path(f'{DATAPATH}/{dir}/target_paper/raw_paper/paper_md/{dir}.md')
            with open(paper_md, 'r', encoding='utf-8') as f:
                target_paper_content = f.read()
        
            get_target_paper_idea(
                target_paper_md=target_paper_content,
                output_path=Path(f'{DATAPATH}/{dir}/target_paper/raw_paper/paper_idea.json')
            )
            
            # exit()
        except Exception as e:
            print(e)
            continue
