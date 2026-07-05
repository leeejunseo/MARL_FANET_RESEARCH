PYTHON=python

.PHONY: all train eval plots test demo run

all: run

train:
	$(PYTHON) train.py

eval:
	$(PYTHON) eval.py

plots:
	$(PYTHON) utils/generate_all_plots.py

test:
	$(PYTHON) test.py

demo:
	$(PYTHON) visualize_attack.py --policy trained --demo-60 --demo-seconds 60 --fps 3 --output logs/demo_60s.gif --event-log logs/demo_60s_link_events.csv --no-show

run: train eval plots demo
