PYTHON=python

.PHONY: all train eval plots test run

all: run

train:
	$(PYTHON) train.py

eval:
	$(PYTHON) eval.py

plots:
	$(PYTHON) utils/generate_all_plots.py

test:
	$(PYTHON) test.py

run: train eval plots
