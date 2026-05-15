#!/bin/bash
set -e
cd "$(dirname "$0")"
python generate_figures.py
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex
echo "report.pdf generated"
