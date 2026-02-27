#!/usr/bin/env python3


# Standard library imports
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Third-party library imports
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Add parent directory to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project-specific imports
from data_prepare.utils import load_config, call_agent, Paper_Profile, Paper

# Load environment variables from .env file (API keys, base URLs, etc.)
load_dotenv()




def step2_paper_analysis(
    all_papers: list[Paper] = None,
    paper_profile: Paper_Profile = None,
    split_ratio: float = 0.5,
    generate_idea_sparks: bool = True
):
    """

    """
    # Display processing configuration
    print("🧠 Testing Enhanced Paper Memory and Idea Spark System")
    print(f"📊 Split ratio: {split_ratio*100:.0f}/{(1-split_ratio)*100:.0f}")
    print(f"💡 Idea sparks: {'Enabled' if generate_idea_sparks else 'Disabled'}\n")
    print(f"Total papers: {len(all_papers)}")
    now = datetime.now().strftime("%Y-%m-%d")
    if paper_profile is None:
        # print(f"Creating empty paper profile")
        exit(f"Paper profile is None")
    # else:
    #     print(f"Reading paper memory from {paper_profile.memory_path}")
    #     paper_profile.read_paper_memory()
    #     print(f"Reading paper group from {paper_profile.group_path}")
    #     paper_profile.read_paper_group()


    config = load_config()
    
    for this_paper in all_papers:
        
        this_paper_id = this_paper.get_paper_id()
        this_paper_memory = this_paper.get_paper_memory()
        this_paper_content = this_paper.get_paper_content()
        SAVE = paper_profile.add_paper_memory(this_paper_memory)
        if not SAVE:
            print(f"this paper: {this_paper_memory['paper_id']} already saved in paper profile!")
            continue
        print("=" * 60)
        print(f"Paper {this_paper.paper_id}: Processing {this_paper.paper_id}")
        print("=" * 60 + "\n")
        
        # ===== Phase 1: Paper Routing =====
        print("🔀 Phase 1: Routing papers to existing memories...")

        # Create index of existing papers for routing context
        paper_memory = [
            m for m in paper_profile.paper_memory["memories"]
        ]
        paper_group = paper_profile.paper_group
        
        
        # Load paper router configuration and format prompt
        router_config = config['agents']['paper_router']
        router_prompt = router_config['prompt'].format(
            paper_memory=json.dumps(paper_group, indent=2, ensure_ascii=False),
            paper_content=json.dumps(this_paper_content, indent=2, ensure_ascii=False),
            paper_info = json.dumps(this_paper_memory, indent=2, ensure_ascii=False),
            schema=json.dumps(
                OmegaConf.to_container(router_config['schema'], resolve=True),
                indent=2,
                ensure_ascii=False
            )
        )

        # Call paper router agent to determine routing decisions
        routing = call_agent('paper_router', router_prompt, config)

        # Extract routing results
        extractions = routing.get('extractions', [])      # Papers to process
        filtered_out = routing.get('filtered_out', [])    # Papers to ignore
        merges = routing.get('merges', [])                # Paper merge operations

        print(f"✅ Routing completed:")
        print(f"   - {len(merges)} paper merges")
        print(f"   - {len(extractions)} paper extractions")
        print(f"   - {len(filtered_out)} papers filtered out\n")

        # ===== Phase 2: Process Extractions =====
        print("🔄 Phase 2: Processing paper extractions...")

        # Aggregate updates by target paper and collect creator tasks
        updates_tasks = []
        creator_tasks = []

        for extraction in extractions:
            target_paper_ids = extraction['paper_ids']  # Simplified paper IDs from router
            decision = extraction.get('decision')
            target_memory_ids = extraction.get('target_memory_ids', [])
            reasoning = extraction.get('reasoning', '')


            # Route based on router decision
            match decision:
                case 'update_existing':
                    print(f"  ↻ update_existing")
                    print(f"     Target papers: {', '.join(target_paper_ids)}")
                    print(f"     Reasoning: {reasoning}")

                    updates_tasks.append({
                        'reasoning': reasoning,
                        'paper_content': this_paper_content,
                        'paper_memory': this_paper_memory,
                        'target_memory_ids': target_memory_ids
                    })

                case 'create_new':
                    print(f"  + create_new")
                    print(f"     Reasoning: {reasoning}")

                    # Collect papers for new memory creation
                    creator_tasks.append({
                        'reasoning': reasoning,
                        'paper_content': this_paper_content,
                        'paper_memory': this_paper_memory
                    })

        # Process paper updates
        if updates_tasks:
            print(f"\n  📝 Calling paper_updater for {len(updates_tasks)} papers...")
            for task in updates_tasks:
                print(f"Task: {task.keys()}")
                this_paper_memory = task['paper_memory']
                create_reason = task['reasoning']
                update_config = config['agents']['paper_group_updater']
                
                update_prompt = update_config['prompt'].format(
                    paper_analysis=json.dumps(this_paper_memory, indent=2, ensure_ascii=False),
                    existing_memories=json.dumps(paper_profile.paper_group, indent=2, ensure_ascii=False),
                    schema=json.dumps(
                        OmegaConf.to_container(update_config['schema'], resolve=True),
                        indent=2,
                        ensure_ascii=False
                    )
                )
                UPDATE_FLAG = False
                while not UPDATE_FLAG:
                    result = call_agent('paper_group_updater', update_prompt, config)
                    result['group_content'] = this_paper_memory
                    UPDATE_FLAG = paper_profile.update_paper_group(result)
               
        # Process new paper creation
        if creator_tasks:
            print(f"\n  🆕 Calling paper_creator for {len(creator_tasks)} new papers...")

            for task in creator_tasks:
                print(f"Task: {task.keys()}")
                this_paper_memory = task['paper_memory']
                create_reason = task['reasoning']
                creator_config = config['agents']['paper_group_creator']
                
                creator_prompt = creator_config['prompt'].format(
                    paper_analysis=json.dumps(this_paper_memory, indent=2, ensure_ascii=False),
                    existing_memories=json.dumps(paper_profile.paper_group, indent=2, ensure_ascii=False),
                    schema=json.dumps(
                        OmegaConf.to_container(creator_config['schema'], resolve=True),
                        indent=2,
                        ensure_ascii=False
                    )
                )

                result = call_agent('paper_group_creator', creator_prompt, config)
                result['group_content'] = [this_paper_memory]
                paper_profile.add_paper_group(result)
                
                
    print(f"\n✅ Batch completed, profile now has {len(paper_profile.paper_memory['papers'])} papers\n")
    return paper_profile
   
if __name__ == "__main__":
    DATAPATH = 'test'

    paper_memory_path = f"{DATAPATH}/layer0/paper_memory"
    paper_md_path = f"{DATAPATH}/raw_paper/paper_md"

    all_paper_ids = []
    for paper_id in os.listdir(paper_md_path):
        if paper_id.endswith('.md'):
            paper_id = paper_id[:-3]
            all_paper_ids.append(paper_id)
    all_papers = []
    for paper_id in all_paper_ids:
        this_paper = Paper(paper_id=paper_id,paper_md_path=paper_md_path,paper_memory_path=paper_memory_path)
        all_papers.append(this_paper)
    paper_profile = Paper_Profile(memory_path=f'{DATAPATH}/layer1/inner_paper_memory.json', group_path=f'{DATAPATH}/layer1/inter_paper_group.json', create_empty=True)
    # Test with papers from data directory
    result = step2_paper_analysis(
        all_papers=all_papers,
        paper_profile=paper_profile,
        split_ratio=0.5,
        generate_idea_sparks=True
    )
