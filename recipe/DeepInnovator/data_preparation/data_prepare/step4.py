#!/usr/bin/env python3


# Standard library imports
import json
import os
import random
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


class Layer2():
    def __init__(self, json_dir: str | os.PathLike):
        self.json_dir = Path(json_dir)
        if not self.json_dir.exists():
            raise FileNotFoundError(f"Path does not exist: {self.json_dir}")
        if not self.json_dir.is_dir():
            raise NotADirectoryError(f"Expected directory: {self.json_dir}")

        self.json_files = sorted(self.json_dir.glob("*.json"))
        self.data = self._load_all_json()

    def _load_all_json(self) -> dict:
        aggregated = {}
        for file_path in self.json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    aggregated[file_path.name] = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Unable to parse JSON: {file_path}") from exc
        return aggregated

    def info(self) -> None:
        """Print directory, file count, and content summary of each JSON."""
        print(f"Directory: {self.json_dir}")
        print(f"JSON file count: {len(self.json_files)}")
        for filename, content in self.data.items():
            print(f"- {filename}:")
            print(f"keys: {content.keys()}")
    def get_connections(self) -> dict:
        try:
            return self.data['connections.json']['metadata']
        except KeyError:
            print(f"connections.json does not exist")
            print(f"keys: {self.data.keys()}")
            return None
    def get_serendipity(self) -> dict:
        try:
            return self.data['serendipity.json']['metadata']
        except KeyError:
            print(f"serendipity.json does not exist")
            print(f"keys: {self.data.keys()}")
            return None

    def get_research_trending(self) -> dict:
        try:
            return self.data['research_trending.json']['metadata']
        except KeyError:
            print(f"research_trending.json does not exist")
            print(f"keys: {self.data.keys()}")
            return None
    

def dropout_layer2(
    connections: dict,
    serendipity: dict,
    research_trending: dict,
    k: int = 1,
    rng: random.Random | None = None
) -> dict | None:
    """Randomly remove k connections and their associated serendipity/trending items."""
    if connections is None:
        return None
    conn_list = connections.get('connections', [])
    if not conn_list:
        return None

    rng = rng or random
    original_conn_size = len(conn_list)
    k = max(0, min(k, original_conn_size))
    if k == 0:
        return None

    removed_connections: list[dict] = []
    removed_sources: set[str] = set()

    for iteration in range(k):
        removed_idx = rng.randrange(len(conn_list))
        removed_connection = conn_list.pop(removed_idx)
        removed_connections.append(removed_connection)
        removed_sources.update(removed_connection.get('source_papers', []))

    def _filter_by_sources(items: list[dict], label: str) -> list[dict]:
        kept: list[dict] = []
        removed_items: list[tuple[int, dict]] = []
        for idx, item in enumerate(items):
            src = set(item.get('source_papers', []))
            if removed_sources and src and src.issubset(removed_sources):
                removed_items.append((idx, item))
                continue
            kept.append(item)

        for idx, item in removed_items:
            title = item.get('title') or item.get('trend_name')
        return kept

    if serendipity is not None:
        ser_items = serendipity.get('serendipities', [])
        serendipity['serendipities'] = _filter_by_sources(ser_items, 'serendipity')

    if research_trending is not None:
        trend_items = research_trending.get('trending_signals', [])
        research_trending['trending_signals'] = _filter_by_sources(trend_items, 'trending')
    return {
        'removed_sources': list(removed_sources),
        'remaining_connections': len(conn_list),
        
    }



def step4_idea_spark(
    layer1: Paper_Profile = None,
    layer2: Layer2 = None,
    DATAPATH: str = None,
    dropout_ratio: float = 0.2,
    min_ideas: int = 10,
    max_iterations: int = 20
):
    config = load_config()
  


    ideas = {'ideas': []}
    agent_config = config['agents']['paper_idea_spark']
    
    iteration = 0
    while len(ideas['ideas']) < min_ideas and iteration < max_iterations:
        connections = layer2.get_connections()
        serendipity = layer2.get_serendipity()
        research_trending = layer2.get_research_trending()

        paper_memories = layer1.paper_memory['memories']
        paper_groups = layer1.paper_group

        dropout_k = int(len(connections['connections']) * dropout_ratio)

        if dropout_k == 0:
            dropout_k = 1
        dropout_summary = dropout_layer2(
            connections,
            serendipity,
            research_trending,
            k=dropout_k
        )
        dropout_id = dropout_summary['removed_sources']



        agent_prompt = agent_config['prompt'].format(
            paper_groups=json.dumps(paper_groups, indent=2, ensure_ascii=False),
            paper_memories=json.dumps(paper_memories, indent=2, ensure_ascii=False),
            connections=json.dumps(connections, indent=2, ensure_ascii=False),
            serendipity=json.dumps(serendipity, indent=2, ensure_ascii=False),
            research_trending=json.dumps(research_trending, indent=2, ensure_ascii=False),
            schema=json.dumps(
                OmegaConf.to_container(agent_config['schema'], resolve=True),
                indent=2,
                ensure_ascii=False
            )
        )
    
        result = call_agent('paper_idea_spark', agent_prompt, config)

        for i in result['ideas']:
            ideas['ideas'].append(i)
        
        iteration += 1
        print(f"Iteration {iteration}: Currently collected {len(ideas['ideas'])} ideas, target: {min_ideas}")

    if not os.path.exists(f'{DATAPATH}/insights'):
        os.makedirs(f'{DATAPATH}/insights')
    with open(f'{DATAPATH}/insights/idea_spark.json', 'w', encoding='utf-8') as f:
        json.dump(ideas, f, indent=2, ensure_ascii=False)
    
    print(f"Finally collected {len(ideas['ideas'])} ideas")
    return ideas


if __name__ == "__main__":
    DATAPATH = 'test'
    layer1 = Paper_Profile(memory_path=f'{DATAPATH}/layer1/inner_paper_memory.json', group_path=f'{DATAPATH}/layer1/inter_paper_group.json', create_empty=False)
    layer2 = Layer2(json_dir=f'{DATAPATH}/layer2')
    
    # Test with papers from data directory
    ideas = step4_idea_spark(
        layer1=layer1,
        layer2=layer2,
        DATAPATH=DATAPATH
    )
