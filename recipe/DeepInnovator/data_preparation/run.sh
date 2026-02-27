#!/bin/bash

# Data Preparation Pipeline Runner Script
# Runs the complete pipeline: download papers -> extract ideas -> generate training data
#
# Usage:
#   bash run.sh [total_papers] [datapath]
#   bash run.sh 100 ./data/arxiv_data

set -e

# Parse arguments
TOTAL_PAPERS=${1:-100}
DATAPATH=${2:-"./data/arxiv_data"}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create .env file with your API credentials."
    exit 1
fi

# Ensure config directory exists
if [ ! -d "config" ]; then
    echo "⚠️  config directory not found!"
    exit 1
fi

echo "🚀 Starting data preparation pipeline..."
echo "  Total papers: $TOTAL_PAPERS"
echo "  Data path: $DATAPATH"
echo ""

# Step 1: Download papers
echo "═══════════════════════════════════════════════════════════"
echo "Step 1: Downloading papers from arXiv"
echo "═══════════════════════════════════════════════════════════"
python data_prepare/pull_papers.py --total_papers "$TOTAL_PAPERS" --datapath "$DATAPATH"

if [ $? -ne 0 ]; then
    echo "❌ Step 1 failed. Exiting."
    exit 1
fi

echo ""
echo "✅ Step 1 completed"
echo ""

# Step 2: Extract target paper ideas
echo "═══════════════════════════════════════════════════════════"
echo "Step 2: Extracting target paper ideas"
echo "═══════════════════════════════════════════════════════════"
python data_prepare/get_target_paper_idea.py --datapath "$DATAPATH"

if [ $? -ne 0 ]; then
    echo "❌ Step 2 failed. Exiting."
    exit 1
fi

echo ""
echo "✅ Step 2 completed"
echo ""

# Step 3: Generate training data
echo "═══════════════════════════════════════════════════════════"
echo "Step 3: Generating training data"
echo "═══════════════════════════════════════════════════════════"
python data_prepare/get_training_data.py

if [ $? -ne 0 ]; then
    echo "❌ Step 3 failed. Exiting."
    exit 1
fi

echo ""
echo "✅ Step 3 completed"
echo ""
echo "🎉 All steps completed successfully!"
echo "   Training data saved in: $DATAPATH"
