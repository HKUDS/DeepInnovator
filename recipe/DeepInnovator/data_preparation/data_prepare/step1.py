#!/usr/bin/env python3

# Standard library imports
import json
import os
import sys
from datetime import datetime
from itertools import count as count_iterator
from pathlib import Path

# Third-party library imports
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Add parent directory to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project-specific imports
from data_prepare.utils import load_config, call_agent, setup_logger

# Load environment variables from .env file (API keys, base URLs, etc.)
load_dotenv()

def format_paper_content(paper: dict, now: str, counter) -> tuple[str, dict[str, str]]:
    """
    Format raw paper content into a standardized JSON Lines format suitable for LLM processing.

    This function transforms a single paper dictionary into a structured format that can be
    consumed by AI agents while maintaining a mapping between internal simplified IDs and
    the original paper identifiers. The output is designed to be compatible with the
    agent communication protocol.

    Args:
        paper (dict): Paper dictionary containing paper metadata and content with keys:
            - 'paper_id' (str): Unique identifier for the paper (typically filename stem)
            - 'title' (str): Paper title for display and reference
            - 'content' (str or list): Full paper content as text or list of page segments
            - 'date' (str): Publication or analysis date in YYYY-MM-DD format
            - Additional optional metadata fields

        now (str): Current date string in "YYYY-MM-DD" format used as fallback when
                   paper date is not available

        counter (iterator): Counter iterator for generating sequential simplified paper IDs
                           (e.g., paper_1, paper_2, paper_3, ...). This ensures consistent
                           numbering across multiple runs and batches.

    Returns:
        tuple[str, dict[str, str]]: A tuple containing:
            - formatted_content (str): JSON Lines formatted string ready for LLM consumption
            - id_mapping (dict): Dictionary mapping simplified model IDs (paper_X) to
                                original paper IDs for traceability and post-processing

    Design Rationale:
        - Uses simplified IDs (paper_1, paper_2, etc.) to provide clean, consistent identifiers
          for the LLM without exposing potentially complex original IDs
        - Maintains bidirectional mapping to enable correlation between model outputs and
          original papers during result processing
        - Formats as JSON Lines to support batch processing while keeping individual paper
          boundaries clear
        - Includes fallback date handling to ensure all papers have temporal context
        - Preserves original content structure (whether string or list) for flexibility

    Note:
        This function processes exactly one paper at a time. For batch processing, call this
        function iteratively for each paper in the collection.
    """
    formatted = []
    id_map = {}

    # Generate simplified sequential ID for model processing (paper_1, paper_2, etc.)
    # This provides clean, predictable identifiers for the LLM while maintaining order
    paper_id = f'paper_{next(counter)}'

    # Extract the real/original paper ID from the input, with fallback generation
    # if not provided (though this should rarely happen in normal operation)
    real_paper_id = paper.get('paper_id', f'paper_{len(id_map)}')

    # Create bidirectional mapping: simplified ID -> real paper ID
    # This enables traceability from model outputs back to original source papers
    id_map[paper_id] = real_paper_id

    # Extract paper content with safe default handling
    # Content may be provided as either a string or list of page segments
    content = paper.get('content', '')

    # Format as JSON object for model processing
    # Note: Currently treats content as-is; commented code shows potential list handling
    # if isinstance(content, list):
    #     content = '<--- Page Split --->'.join(content)
    formatted.append(json.dumps({
        'paper_id': paper_id,           # Simplified sequential ID for model consumption
        'title': paper.get('title', ''), # Paper title (with empty string fallback)
        'content': content,             # Full paper content in original format
        'date': paper.get('date', now)   # Paper date with current date as fallback
    }, ensure_ascii=False) + "\n")

    # Return concatenated formatted string (JSON Lines) and ID mapping dictionary
    # The formatted string contains exactly one JSON object followed by newline
    return "".join(formatted), id_map

def step1_paper2json(
    paper_files: list[str] = None,
    input_jsonl_path: str = "data/paper_input.jsonl",
    paper_memory_dir: str = "data/paper_preprocessed",
    logger=None
) -> None:
    # Ensure the paper memory directory exists for output storage
    if not os.path.exists(paper_memory_dir):
        os.makedirs(paper_memory_dir, exist_ok=True)

    # Log the start of Step 1 processing if logger is available
    if logger:
        logger.info("Step 1: Paper Analysis. Generate core ideas from each papers")
        logger.info('=' * 50)

    # Load system configuration (models, agents, structures)
    config = load_config()

    # Initialize data structures for tracking papers
    all_paper_ids = []  # List of all paper IDs from existing JSONL records
    papers = []         # List of all paper dictionaries (existing + new)

    # Load existing papers from JSONL file if it exists (for incremental processing)
    if os.path.exists(input_jsonl_path):
        print('Loading existing paper information...')
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                all_paper_ids.append(paper.get('paper_id', ''))
                papers.append(paper)

    # Process new paper files if provided
    if paper_files:
        new_papers = []
        # Load papers from file paths
        for paper_file in paper_files:
            paper_path = Path(paper_file)
            if paper_path.exists():
                # Read raw paper content from Markdown file
                with open(paper_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split content by page separator (used in paper preprocessing)
                    content_pages = content.split('<--- Page Split --->')

                    # Skip papers that have already been processed (deduplication)
                    if paper_path.stem in all_paper_ids:
                        print(f"❌ Paper {paper_path.stem} already processed")
                        continue

                    # Create paper metadata dictionary with standardized fields
                    new_papers.append({
                        'paper_id': paper_path.stem,  # Use filename stem as unique ID
                        'title': paper_path.stem.replace('_', ' ').title(),  # Clean and format title
                        'content': content_pages,  # Content as list of page segments
                        'total_pages': len(content_pages),  # Number of pages for reference
                        'date': datetime.now().strftime("%Y-%m-%d"),  # Current date as processing date
                        'paper_path': str(paper_path),  # Original file path for traceability
                    })
            else:
                # Handle missing paper files gracefully
                print(f"❌ Paper file not found: {paper_path}")
                return None
    else:
        # Exit if no input files are provided
        print('No input!')
        exit()

    # Persist new paper metadata to JSONL file for incremental processing support
    os.makedirs(os.path.dirname(input_jsonl_path), exist_ok=True)
    with open(input_jsonl_path, 'a', encoding='utf-8') as f:
        for paper in new_papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        print(f"✅ Saved {len(new_papers)} new paper information to {input_jsonl_path}\nOriginally had {len(papers)} papers")

    # Combine existing and new papers for processing
    papers.extend(new_papers)

    # Filter out papers that have already been processed (based on output JSON files)
    # This provides an additional layer of deduplication beyond the JSONL check
    existing_titles = [f.stem for f in Path(paper_memory_dir).glob("*.json")]
    papers = [paper for paper in papers if paper.get('title', '').lower().strip() not in existing_titles]
    print(f"Filtered papers count: {len(papers)}, originally had {len(all_paper_ids)} papers")

    # Initialize timestamp and counter for paper processing
    now = datetime.now().strftime("%Y-%m-%d")
    # Counter starts from appropriate offset to maintain sequential numbering
    paper_counter = count_iterator(len(all_paper_ids) - len(papers))

    # Process each paper sequentially through the AI analysis pipeline
    for paper_idx, this_paper in enumerate(papers):
        print("=" * 60)
        print(f"Paper {paper_idx}/{len(papers)}: Processing {this_paper.get('title', '')}")
        print("=" * 60 + "\n")

        # Format paper content for LLM consumption and create ID mapping
        papers_formatted, paper_id_map = format_paper_content(this_paper, now, paper_counter)
        this_paper_id = this_paper['paper_id']
        this_paper_path = this_paper['paper_path']

        # ===== Phase 1: Paper Analysis =====
        print("🔀 Phase 1: Paper Analysis...")

        # Load paper analysis agent configuration and format the prompt
        router_config = config['agents']['paper_analyzer']
        router_prompt = router_config['prompt'].format(
            paper_content=json.dumps([papers_formatted], indent=2, ensure_ascii=False),
            schema=json.dumps(
                OmegaConf.to_container(router_config['schema'], resolve=True),
                indent=2,
                ensure_ascii=False
            )
        )

        # Call the paper analyzer agent to extract structured information
        # Note: The function calls 'paper_router' but the config references 'paper_analyzer'
        # This may be a naming inconsistency that should be addressed
        routing = call_agent('paper_analyzer', router_prompt, config)

        # Create structured output dictionary with extracted information
        save_dict = {
            'paper_id': this_paper_id,
            'paper_title': routing.get('paper_title', ''),
            'paper_summary': routing.get('paper_summary', ''),
            'research_domain': routing.get('research_domain', ''),
            'key_findings': routing.get('key_findings', []),
            'methodology': routing.get('methodology', ''),
            'limitations': routing.get('limitations', []),
            'future_work': routing.get('future_work', []),
            'confidence': routing.get('confidence', 0.0),
            'paper_path': this_paper_path
        }

        # Persist the analyzed paper to JSON file in memory directory
        with open(os.path.join(paper_memory_dir, f"{this_paper_id}.json"), 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ Paper analysis completed: {save_dict}")
        
        

if __name__ == "__main__":
    DATAPATH = 'test'
    print("Step 1: Paper Analysis. Generate core ideas from each papers")
    print("=" * 50)

    # Configure default paths for paper processing
    paper_path = Path(f'{DATAPATH}/raw_paper/paper_md')      # Directory containing raw paper Markdown files
    log_path = Path('logs/paper_step1.log') # Log file path for execution monitoring

    # Ensure log directory exists
    os.makedirs(log_path.parent, exist_ok=True)

    # Clear existing log file content to start fresh (useful for testing/debugging)
    # In production, you might want to append to logs instead of clearing
    if log_path.exists() and log_path.stat().st_size > 0:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('')

    # Initialize logger for comprehensive execution logging
    # The logger is used by the step1_paper2json function for detailed monitoring
    logger = setup_logger(log_file=log_path, logger_name='paper_analysis')

    # Log the start of execution with logger as well as console output
    logger.info("Step 1: Paper Analysis. Generate core ideas from each papers")
    logger.info('=' * 50)

    # Execute the main paper analysis pipeline with default parameters
    # - paper_files: All Markdown files in the data/paper_md directory
    # - input_jsonl_path: Persistent storage for paper metadata records
    # - paper_memory_dir: Output directory for structured paper analysis results
    # - logger: Logger instance for comprehensive execution logging
    step1_paper2json(
        paper_files=list(paper_path.glob("*.md")),
        input_jsonl_path=f"{DATAPATH}/raw_paper/paper_input.jsonl",
        paper_memory_dir=f"{DATAPATH}/layer0/paper_memory",
        logger=logger
    )

