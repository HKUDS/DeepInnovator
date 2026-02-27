#!/usr/bin/env python3
"""
Batch download papers from arXiv

Usage:
    python pull_100_papers.py --total_papers <number> --datapath <path>
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


# === Path and dependency setup ===
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prepare.utils import ArxivFetcher, RawPaper  # noqa: E402

# Target: Start fetching from 2025-03-01 onwards
SINCE_DATE_STR = "2025-03-01"
SINCE_DATE = datetime.strptime(SINCE_DATE_STR, "%Y-%m-%d").date()

# Predefined categories: 4 major categories, 5 subcategories each (20 total)
TARGET_CATEGORIES: List[str] = [
    # Computer Science (5)
    "cs.AI",  # Artificial Intelligence
    "cs.LG",  # Machine Learning
    "cs.CL",  # Computation and Language
    "cs.CV",  # Computer Vision
    "cs.IR",  # Information Retrieval
    
    # Statistics (5)
    "stat.ML",  # Machine Learning
    "stat.AP",  # Applications
    "stat.CO",  # Computation
    "stat.TH",  # Statistics Theory
    "stat.ME",  # Methodology
    
    # Quantitative Finance (5)
    "q-fin.TR",  # Trading and Market Microstructure
    "q-fin.RM",  # Risk Management
    "q-fin.PM",  # Portfolio Management
    "q-fin.ST",  # Statistical Finance
    "q-fin.EC",  # Econometrics
    
    # Mathematics (5)
    "math.OC",  # Optimization and Control
    "math.PR",  # Probability
    "math.ST",  # Statistics Theory
    "math.NA",  # Numerical Analysis
    "math.AT",  # Algebraic Topology
]


def fetch_recent_papers_for_category(
    category: str,
    max_results: int,
) -> List[Dict]:
    """
    Use ArxivFetcher to fetch papers by category + date range.
    
    Args:
        category: Category name (e.g., "cs.AI")
        max_results: Maximum number of papers to fetch
    
    Returns:
        List of paper dictionaries
    """
    fetcher = ArxivFetcher(page_size=15)
    
    start = SINCE_DATE.strftime("%Y%m%d0000")
    end = "210001012359"
    date_clause = f"submittedDate:[{start} TO {end}]"
    query = f"({date_clause}) AND cat:{category}"
    
    print(f"Fetching papers for category {category} with query: {query}")
    papers = fetcher._fetch_with_query(search_query=query, max_results=max_results)  # type: ignore[attr-defined]
    print(f"Fetched {len(papers)} papers for category {category}")
    return papers


def download_papers(papers: List[Dict], datapath: Path) -> None:
    """
    Download papers to specified datapath.
    
    Directory structure:
    - {datapath}/raw_paper/original_paper/*.pdf
    - {datapath}/raw_paper/paper_json/*.json  (OCR results)
    - {datapath}/raw_paper/paper_md/*.md      (Markdown content)
    
    Args:
        papers: List of paper dictionaries
        datapath: Data save path
    """
    save_path = datapath / "raw_paper" / "original_paper"
    ocr_save_path = datapath / "raw_paper" / "paper_json"
    md_save_path = datapath / "raw_paper" / "paper_md"
    
    for idx, paper_dict in enumerate(papers, 1):
        paper = RawPaper(paper_dict)
        filename = paper.extract_filename()
        if not filename:
            print(f"Warning: Unable to extract filename from ID: {paper.id}")
            continue
        
        print(f"({idx}/{len(papers)}) Processing paper: {paper.title[:80] if paper.title else 'N/A'} | ID: {paper.id} | Filename: {filename}.pdf")
        try:
            paper.download_and_extract(
                save_path=save_path,
                ocr_save_path=ocr_save_path,
                md_save_path=md_save_path,
                filename=filename,
                auto_ocr=True,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"Error: Failed to download/process paper: {exc}")


def main() -> None:
    """
    Main function: Download papers from predefined categories.
    
    Categories: cs (5), stat (5), q-fin (5), math (5) - total 20 subcategories
    Papers are evenly distributed across all subcategories.
    """
    parser = argparse.ArgumentParser(description="Batch download papers from arXiv")
    parser.add_argument(
        "--total_papers",
        type=int,
        required=True,
        help="Total number of papers to download"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Data save path"
    )
    
    args = parser.parse_args()
    
    total_papers = args.total_papers
    datapath = Path(args.datapath)
    datapath.mkdir(parents=True, exist_ok=True)
    
    # Calculate papers per category (evenly distributed)
    papers_per_category = max(1, total_papers // len(TARGET_CATEGORIES))
    
    print(f"Starting to fetch papers:")
    print(f"  Target: {total_papers} papers")
    print(f"  Categories: {len(TARGET_CATEGORIES)} subcategories")
    print(f"  Papers per category: ~{papers_per_category}")
    print(f"  Since date: {SINCE_DATE_STR}")
    print(f"  Save path: {datapath}")
    
    all_papers: List[Dict] = []
    seen_ids: Set[str] = set()
    # Batch download cache: download + OCR every 10 papers
    batch_papers: List[Dict] = []
    
    # Fetch papers from each category
    for idx, category in enumerate(TARGET_CATEGORIES, 1):
        print(f"\nProcessing category {idx}/{len(TARGET_CATEGORIES)}: {category}")
        
        # Calculate how many papers we still need
        remaining = total_papers - len(all_papers)
        if remaining <= 0:
            break
        
        # Fetch more papers than needed to account for duplicates or invalid papers
        papers_to_fetch = min(papers_per_category * 2, remaining * 2)
        cat_papers = fetch_recent_papers_for_category(
            category, max_results=papers_to_fetch
        )
        
        papers_added = 0
        for paper in cat_papers:
            pid = paper.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            all_papers.append(paper)
            batch_papers.append(paper)
            papers_added += 1
            
            # Download + OCR when batch reaches 10 papers
            if len(batch_papers) >= 10:
                print(f"  Batch full (10 papers), starting download + OCR. Current progress: {len(all_papers)} / {total_papers}")
                download_papers(batch_papers, datapath=datapath)
                batch_papers.clear()
            
            # Stop if we've reached the target
            if len(all_papers) >= total_papers:
                break
            
            # Stop if we've added enough papers for this category
            if papers_added >= papers_per_category:
                break
        
        print(f"  Category {category}: Added {papers_added} papers, current total: {len(all_papers)} / {total_papers}")
        
        if len(all_papers) >= total_papers:
            break
    
    # Process remaining papers (less than 10 in the last batch)
    if batch_papers:
        print(f"\nProcessing final batch: {len(batch_papers)} papers (less than 10)")
        download_papers(batch_papers, datapath=datapath)
        batch_papers.clear()
    
    print(f"\n✓ All processing complete. Processed {len(all_papers)} papers in total")
    print(f"Data saved in: {datapath}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("User interrupted script execution")
        sys.exit(1)


