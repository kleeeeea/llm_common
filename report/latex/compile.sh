#!/usr/bin/env bash
# Compile main.tex → main.pdf  (pdflatex, two passes for cross-references)
set -euo pipefail
cd "$(dirname "$0")"
echo "==> Pass 1: pdflatex"
pdflatex -interaction=nonstopmode main.tex
echo "==> Pass 2: pdflatex"
pdflatex -interaction=nonstopmode main.tex
echo "==> Done: main.pdf"
