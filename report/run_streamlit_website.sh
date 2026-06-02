#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="$SCRIPT_DIR/scripts"

cd "$PROJECT_ROOT"

if ! typeset -f conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="${CONDA_EXE:h:h}"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        print -u2 "ERROR: conda is not available on PATH"
        exit 1
    fi

    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda activate base124
exec streamlit run report_gen/streamlit_report.py "$@"
