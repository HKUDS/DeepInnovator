
# Standard library imports
import json
import os
import sys
from pathlib import Path

# Third-party library imports
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file (API keys, base URLs, etc.)
load_dotenv()

# Project-specific imports
from data_prepare.utils import (
    Paper_Profile,
    Paper,
    download_arxiv_papers,
    extract_paper_references
)
from data_prepare.step1 import step1_paper2json
from data_prepare.step2 import step2_paper_analysis
from data_prepare.step3 import step3_paper_analysis
from data_prepare.step4 import step4_idea_spark, Layer2



def prepare_data(DATAPATH: str, target_paper: str):

    target_paper_path_ = Path(f'{DATAPATH}/target_paper/raw_paper/paper_md/')
    download_arxiv_papers([target_paper], f'{DATAPATH}/target_paper', parallel=10)

    target_paper_path = list(target_paper_path_.glob('*.md'))
    print(f"Found {len(target_paper_path)} matching files:")
    ref_path = Path(f'{DATAPATH}/target_paper/raw_paper/paper_references.json')
    if not ref_path.exists():
        for path in target_paper_path:
            all_references = extract_paper_references(path)
            
    with open(ref_path, 'r') as f:
        all_references = json.load(f)
    all_ref_paper = []
    for ref in all_references['references']:
        all_ref_paper.append(ref['url'])
    if len(all_ref_paper) < 5:
        return
    download_arxiv_papers(all_ref_paper, f"{DATAPATH}", parallel=3)

    IDEA_PATH = f"{DATAPATH}/insights/idea_spark.json"
    if os.path.exists(IDEA_PATH):
        print(f"Target paper already has ideas, skipping step1, step2, step3, step4")
        return

    #step1
    print("now step1-------")
    paper_files=list(Path(f'{DATAPATH}/raw_paper/paper_md').glob("*.md"))
    print(f"Total number of papers: {len(paper_files)}")
    if len(paper_files) < 5:
        print(f"Insufficient references, total {len(paper_files)} papers, skipping step1, step2, step3, step4")
        return
    step1_paper2json(
        paper_files=list(Path(f'{DATAPATH}/raw_paper/paper_md').glob("*.md")),
        input_jsonl_path=f'{DATAPATH}/raw_paper/paper_input.jsonl',
        paper_memory_dir=f'{DATAPATH}/layer0/paper_memory',
    )
    #step2
    print("now step2-------")
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
    
    step2_paper_analysis(
        all_papers=all_papers,
        paper_profile=paper_profile,
        split_ratio=0.5,
        generate_idea_sparks=True
    )


    #step3 
    print("now step3-------")
    step3_paper_analysis(DATAPATH)

    #step4
    print("now step4-------")
    layer1 = Paper_Profile(memory_path=f'{DATAPATH}/layer1/inner_paper_memory.json', group_path=f'{DATAPATH}/layer1/inter_paper_group.json', create_empty=False)
    layer2 = Layer2(json_dir=f'{DATAPATH}/layer2')
    # Test with papers from data directory
    step4_idea_spark(
        layer1=layer1,
        layer2=layer2,
        DATAPATH=DATAPATH,
        dropout_ratio=0.2
    )
    return




if __name__ == "__main__":
    user_id = "arxiv_data/"
    subdirs = [name for name in os.listdir(f"data/{user_id}") if os.path.isdir(os.path.join(f"data/{user_id}", name))]
    for dir in tqdm(subdirs):
        try:
            DATAPATH = f'data/{user_id}/{dir}'
            arxiv_id = dir.replace('-', '.').split('_')[-1]
            print(DATAPATH, arxiv_id)
            prepare_data(DATAPATH,arxiv_id)
            with open(f'success.log', 'a') as f:
                f.write(f'{dir} {arxiv_id} success\n')
        except Exception as e:
            with open(f'error.log', 'a') as f:
                f.write(f'{dir} {arxiv_id} {e}\n')
            continue