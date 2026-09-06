# ToySVG: convenience targets for benchmark artifacts.
#
#   make open     open the specification paper (docs/ToySVG_Benchmark_Specification_v0.1.pdf)
#   make test     run task verification checks locally

PYTHON  ?= python3
OUTDIR  := docs
PDFNAME := ToySVG_Benchmark_Specification_v0.1.pdf

.PHONY: all open test help

all: open

open:
	@setsid --fork xdg-open $(OUTDIR)/$(PDFNAME) </dev/null >/dev/null 2>&1 || true

test:
	@$(PYTHON) kaggle/test_task_files.py

help:
	@sed -n '2,6p' Makefile | sed 's/^# \{0,1\}//'
