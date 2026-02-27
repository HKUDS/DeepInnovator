# Data Preparation Pipeline

A multi-step pipeline for processing academic papers from arXiv and generating research ideas.

## Overview

This pipeline processes academic papers through three main stages:

1. **Download Papers**: Fetch papers from arXiv across multiple categories
2. **Extract Ideas**: Extract research ideas from target papers
3. **Generate Training Data**: Process papers through analysis pipeline (4 internal steps) to generate training data

The training data generation includes:
- **Step 1**: Extract structured information from raw paper markdown files
- **Step 2**: Route and group papers into thematic clusters
- **Step 3**: Generate paper connections, serendipity insights, and research trends
- **Step 4**: Synthesize next research ideas based on analyzed papers

## Project Structure

```
data_preparation/
├── config/              # Configuration files
│   ├── agents/         # Agent prompts and schemas
│   └── models/         # Model definitions and model sets
├── data_prepare/       # Main pipeline scripts
│   ├── pull_papers.py         # Step 1: Download papers from arXiv
│   ├── get_target_paper_idea.py  # Step 2: Extract target paper ideas
│   ├── get_training_data.py   # Step 3: Generate training data (calls step1-4)
│   ├── step1.py        # Paper analysis and extraction
│   ├── step2.py        # Paper routing and grouping
│   ├── step3.py        # Connections and insights generation
│   ├── step4.py        # Idea synthesis
│   └── utils.py        # Utility functions
└── README.md           # This file
```

## Prerequisites

- Python 3.8+
- Required packages: `openai`, `omegaconf`, `python-dotenv`, `feedparser`, `requests`, `PyPDF2`, `tqdm`, `dateutil`

## Environment Setup

1. Create `.env` file and configure:
   ```bash
   OPENAI_API_BASE=your_api_base_url
   OPENAI_API_KEY=your_api_key
   ```

2. Install dependencies:
   ```bash
   pip install openai omegaconf python-dotenv feedparser requests PyPDF2 tqdm python-dateutil
   ```

## Usage

### Workflow

The pipeline consists of three main steps:

1. **Download papers from arXiv** (`pull_papers.py`)
2. **Extract target paper ideas** (`get_target_paper_idea.py`)
3. **Generate training data** (`get_training_data.py`)

### Step-by-Step

#### Step 1: Download Papers

Download papers from arXiv across predefined categories (cs, stat, q-fin, math):

```bash
python data_prepare/pull_papers.py --total_papers 100 --datapath ./data/arxiv_data
```

**Parameters:**
- `--total_papers`: Total number of papers to download
- `--datapath`: Data save path (e.g., `./data/arxiv_data`)

**Output:** Papers saved to `{datapath}/raw_paper/` directory

#### Step 2: Extract Target Paper Ideas

Extract ideas from target papers:

```bash
python data_prepare/get_target_paper_idea.py --datapath ./data/arxiv_data
```

**Parameters:**
- `--datapath` or `-d`: Path to arxiv data directory (default: `data/arxiv_data`)

**Output:** `{datapath}/{paper_id}/target_paper/raw_paper/paper_idea.json`

#### Step 3: Generate Training Data

Process papers through the full pipeline (step1-step4) to generate training data:

```bash
python data_prepare/get_training_data.py
```

**Note:** This script processes all subdirectories in `data/arxiv_data/` automatically.

**Output:** 
- `layer0/`: Paper analysis results
- `layer1/`: Paper groups and memories
- `layer2/`: Connections, serendipity, and trends
- `insights/`: Generated research ideas

### Quick Run

Run all steps sequentially:

```bash
# Use default values (100 papers, ./data/arxiv_data)
bash run.sh

# Or specify custom parameters
bash run.sh 200 ./data/my_papers
```

**Parameters:**
- First argument: Total number of papers to download (default: 100)
- Second argument: Data path (default: ./data/arxiv_data)

This will execute all three steps in order.

### Configuration

- Model selection: Edit `config/models/model_sets.yaml`
- Agent prompts: Edit `config/agents/*.yaml`

## Pipeline Steps

### Step 1: Paper Analysis (`step1.py`)
- Extracts structured information from paper markdown files
- Outputs: `layer0/paper_memory/` with JSON analysis files

### Step 2: Paper Routing (`step2.py`)
- Routes papers to existing groups or creates new ones
- Outputs: `layer1/inner_paper_memory.json` and `layer1/inter_paper_group.json`

### Step 3: Connections & Insights (`step3.py`)
- Generates paper connections, serendipity insights, and research trends
- Outputs: `layer2/connections.json`, `layer2/serendipity.json`, `layer2/research_trending.json`

### Step 4: Idea Synthesis (`step4.py`)
- Synthesizes next research ideas using qwen-14b-instruct model
- Outputs: `insights/idea_spark.json`

## Output Structure

```
data/
└── {paper_id}/
    ├── target_paper/          # Target paper and references
    ├── raw_paper/             # Raw downloaded papers
    ├── layer0/                # Step 1 outputs
    ├── layer1/                # Step 2 outputs
    ├── layer2/                # Step 3 outputs
    └── insights/              # Step 4 outputs
```

## Notes

- The pipeline uses LLM agents for each step
- Step 4 uses `qwen-14b-instruct` model (configured in `config/agents/paper_idea_spark.yaml`)
- All steps can be run independently or as a complete pipeline
