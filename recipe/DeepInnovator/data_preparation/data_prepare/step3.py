#!/usr/bin/env python3


# Standard library imports
import json
import os
import sys
from pathlib import Path

# Third-party library imports
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Add parent directory to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project-specific imports
from data_prepare.utils import load_config, call_agent, Paper_Profile

# Load environment variables from .env file (API keys, base URLs, etc.)
load_dotenv()


def generate_paper_connections(
    paper_profile: Paper_Profile = None,
    split_ratio: float = 0.5,
    generate_idea_sparks: bool = True
):
    config = load_config()
  
    print("  🔗 Generating paper connections...")
    connections_config = config['agents']['idea_paper_connections']

    # Prepare papers data for connections agent
    papers_for_connections = paper_profile.paper_memory['memories']
    
    # Format prompt with schema and paper_memories
    connections_prompt = connections_config['prompt'].format(
        paper_memories=json.dumps(papers_for_connections, indent=2, ensure_ascii=False),
        schema=json.dumps(
            OmegaConf.to_container(connections_config['schema'], resolve=True),
            indent=2,
            ensure_ascii=False
        )
    )
    
    # Call paper connections agent
    
    connections_result = call_agent('idea_paper_connections', connections_prompt, config)
    print(connections_result)
    return connections_result
def generate_serendipity_insights(
    paper_profile: Paper_Profile = None,
    split_ratio: float = 0.5,
    generate_idea_sparks: bool = True
):
    config = load_config()
  
    print("  ✨ Generating serendipity insights...")
    serendipity_config = config['agents']['idea_serendipity_engine']

    # Prepare user memories data for serendipity agent
    paper_memories = paper_profile.paper_memory['memories']
    paper_groups = paper_profile.paper_group#['groups']
    # Format prompt with schema and user_memories
    serendipity_prompt = serendipity_config['prompt'].format(
        user_memories=json.dumps(paper_memories, indent=2, ensure_ascii=False),
        schema=json.dumps(
            OmegaConf.to_container(serendipity_config['schema'], resolve=True),
            indent=2,
            ensure_ascii=False
        )
    )

    # Call serendipity engine agent
    serendipity_result = call_agent('idea_serendipity_engine', serendipity_prompt, config)
    return serendipity_result
  
def generate_research_trending(
    paper_profile: Paper_Profile = None,
    split_ratio: float = 0.5,
    generate_idea_sparks: bool = True
):
    config = load_config()
    print("  📊 Generating research trending...")
    research_trending_config = config['agents']['idea_research_trending']
    research_trending_prompt = research_trending_config['prompt'].format(
        paper_memories=json.dumps(paper_profile.paper_memory['memories'], indent=2, ensure_ascii=False),
        schema=json.dumps(
            OmegaConf.to_container(research_trending_config['schema'], resolve=True),
            indent=2,
            ensure_ascii=False
        )
    )
    
    research_trending_result = call_agent('idea_research_trending', research_trending_prompt, config)
    return research_trending_result


def _is_result_empty(result: dict, data_key: str) -> bool:
    """Check if generated result is empty
    
    Args:
        result: Generated result dictionary
        data_key: Key name of data array (e.g., 'connections', 'serendipities', 'trending_signals')
    
    Returns:
        True if result is empty, False otherwise
    """
    if not result or not isinstance(result, dict):
        return True
    if 'metadata' not in result:
        return True
    metadata = result.get('metadata', {})
    data_array = metadata.get(data_key, [])
    return not data_array or len(data_array) == 0


def step3_paper_analysis(DATAPATH: str):

    paper_profile = Paper_Profile(memory_path=f'{DATAPATH}/layer1/inner_paper_memory.json', group_path=f'{DATAPATH}/layer1/inter_paper_group.json', create_empty=False)
    layer2_dir = Path(DATAPATH) / "layer2"
    layer2_dir.mkdir(parents=True, exist_ok=True)
    print(f"Layer2 directory created at {layer2_dir}")

    # paper_profile.info()
    max_retries = 3
    retry_count = 0
    connections_result = None
    while retry_count < max_retries:
        connections_result = generate_paper_connections(
            paper_profile=paper_profile,
            split_ratio=0.5,
            generate_idea_sparks=True
        )
        if not _is_result_empty(connections_result, 'connections'):
            break
        retry_count += 1
        print(f"  ⚠️  Connections result is empty, retrying ({retry_count}/{max_retries})...")
    
    if _is_result_empty(connections_result, 'connections'):
        print(f"  ❌ Failed to generate connections after {max_retries} attempts")
    else:
        connection_path = f'{DATAPATH}/layer2/connections.json'
        with open(connection_path, 'w', encoding='utf-8') as f:
            json.dump(connections_result, f, indent=2, ensure_ascii=False)
        print(f"Connections saved to {connection_path}")

    retry_count = 0
    serendipity_result = None
    while retry_count < max_retries:
        serendipity_result = generate_serendipity_insights(
            paper_profile=paper_profile,
            split_ratio=0.5,
            generate_idea_sparks=True
        )
        if not _is_result_empty(serendipity_result, 'serendipities'):
            break
        retry_count += 1
        print(f"  ⚠️  Serendipity result is empty, retrying ({retry_count}/{max_retries})...")
    
    if _is_result_empty(serendipity_result, 'serendipities'):
        print(f"  ❌ Failed to generate serendipity insights after {max_retries} attempts")
    else:
        serendipity_path = f'{DATAPATH}/layer2/serendipity.json'
        with open(serendipity_path, 'w', encoding='utf-8') as f:
            json.dump(serendipity_result, f, indent=2, ensure_ascii=False)
        print(f"Serendipity insights saved to {serendipity_path}")
    
    retry_count = 0
    research_trending_result = None
    while retry_count < max_retries:
        research_trending_result = generate_research_trending(
            paper_profile=paper_profile,
            split_ratio=0.5,
            generate_idea_sparks=True
        )
        if not _is_result_empty(research_trending_result, 'trending_signals'):
            break
        retry_count += 1
        print(f"  ⚠️  Research trending result is empty, retrying ({retry_count}/{max_retries})...")
    
    if _is_result_empty(research_trending_result, 'trending_signals'):
        print(f"  ❌ Failed to generate research trending after {max_retries} attempts")
    else:
        research_trending_path = f'{DATAPATH}/layer2/research_trending.json'
        with open(research_trending_path, 'w', encoding='utf-8') as f:
            json.dump(research_trending_result, f, indent=2, ensure_ascii=False)
        print(f"Research trending saved to {research_trending_path}")



if __name__ == "__main__":
    DATAPATH = 'test'
    paper_profile = Paper_Profile(memory_path=f'{DATAPATH}/layer1/inner_paper_memory.json', group_path=f'{DATAPATH}/layer1/inter_paper_group.json', create_empty=False)
    paper_profile.info()
    
    max_retries = 3
    retry_count = 0
    connections_result = None
    while retry_count < max_retries:
        connections_result = generate_paper_connections(
            paper_profile=paper_profile,
            split_ratio=0.5,
            generate_idea_sparks=True
        )
        if not _is_result_empty(connections_result, 'connections'):
            break
        retry_count += 1
        print(f"  ⚠️  Connections result is empty, retrying ({retry_count}/{max_retries})...")
    
    if _is_result_empty(connections_result, 'connections'):
        print(f"  ❌ Failed to generate connections after {max_retries} attempts")
    else:
        connection_path = f'{DATAPATH}/layer2/connections.json'
        with open(connection_path, 'w', encoding='utf-8') as f:
            json.dump(connections_result, f, indent=2, ensure_ascii=False)
        print(f"Connections saved to {connection_path}")

    retry_count = 0
    serendipity_result = None
    while retry_count < max_retries:
        serendipity_result = generate_serendipity_insights(
            paper_profile=paper_profile,
            split_ratio=0.5,
            generate_idea_sparks=True
        )
        if not _is_result_empty(serendipity_result, 'serendipities'):
            break
        retry_count += 1
        print(f"  ⚠️  Serendipity result is empty, retrying ({retry_count}/{max_retries})...")
    
    if _is_result_empty(serendipity_result, 'serendipities'):
        print(f"  ❌ Failed to generate serendipity insights after {max_retries} attempts")
    else:
        serendipity_path = f'{DATAPATH}/layer2/serendipity.json'
        with open(serendipity_path, 'w', encoding='utf-8') as f:
            json.dump(serendipity_result, f, indent=2, ensure_ascii=False)
        print(f"Serendipity insights saved to {serendipity_path}")
    
    retry_count = 0
    research_trending_result = None
    while retry_count < max_retries:
        research_trending_result = generate_research_trending(
            paper_profile=paper_profile,
            split_ratio=0.5,
            generate_idea_sparks=True
        )
        if not _is_result_empty(research_trending_result, 'trending_signals'):
            break
        retry_count += 1
        print(f"  ⚠️  Research trending result is empty, retrying ({retry_count}/{max_retries})...")
    
    if _is_result_empty(research_trending_result, 'trending_signals'):
        print(f"  ❌ Failed to generate research trending after {max_retries} attempts")
    else:
        research_trending_path = f'{DATAPATH}/layer2/research_trending.json'
        with open(research_trending_path, 'w', encoding='utf-8') as f:
            json.dump(research_trending_result, f, indent=2, ensure_ascii=False)
        print(f"Research trending saved to {research_trending_path}")