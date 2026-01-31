#!/bin/bash
set -e

echo "🚀 Starting One-Click Compilation Pipeline..."

# 1. Update Analysis Results & Figures
echo "--------------------------------------------------------"
echo "📊 Step 1: Updating Analysis Results & Figures..."
echo "--------------------------------------------------------"

# Ensure directories exist
mkdir -p results/figures

# Run scripts
echo "Minimizing DCI Dimensionality (PCA)..."
python3 -m scripts.pca_diagnostics

echo "Running Monte Carlo Power Analysis..."
# Set backend to Agg to avoid display issues
export MPLBACKEND=Agg 
export MPLCONFIGDIR=/tmp
python3 -m scripts.power_analysis

echo "Generating Publication Figures..."
python3 -m scripts.phase3_visualizations

echo "✅ Analysis artifacts updated."

# 2. Compile LaTeX
echo "--------------------------------------------------------"
echo "📄 Step 2: Compiling Paper (Chinese Version)..."
echo "--------------------------------------------------------"

if command -v xelatex &> /dev/null; then
    # Cleanup aux files
    rm -f *.aux *.log *.out *.toc *.bbl *.blg

    echo "Pass 1: xelatex..."
    xelatex -interaction=nonstopmode paper_cn.tex > /dev/null
    
    echo "Pass 2: bibtex..."
    bibtex paper_cn > /dev/null
    
    echo "Pass 3: xelatex (linking refs)..."
    xelatex -interaction=nonstopmode paper_cn.tex > /dev/null
    
    echo "Pass 4: xelatex (finalizing)..."
    xelatex -interaction=nonstopmode paper_cn.tex > /dev/null
    
    echo "✅ PDF Generated Successfully: paper_cn.pdf"
    
else
    echo "⚠️  xelatex command not found. Please install TeX Live / MacTeX."
fi

echo "--------------------------------------------------------"
echo "📄 Step 3: Compiling Paper (English Version)..."
echo "--------------------------------------------------------"

if command -v pdflatex &> /dev/null; then
    echo "Pass 1: pdflatex..."
    pdflatex -interaction=nonstopmode paper.tex > /dev/null
    
    echo "Pass 2: bibtex..."
    bibtex paper > /dev/null
    
    echo "Pass 3: pdflatex (linking refs)..."
    pdflatex -interaction=nonstopmode paper.tex > /dev/null
    
    echo "Pass 4: pdflatex (finalizing)..."
    pdflatex -interaction=nonstopmode paper.tex > /dev/null
    
    echo "✅ PDF Generated Successfully: paper.pdf"
else
    echo "⚠️  pdflatex command not found."
fi

echo "--------------------------------------------------------"
echo "🎉 Pipeline Complete! Files are ready."
echo "--------------------------------------------------------"
