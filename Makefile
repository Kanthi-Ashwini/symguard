# SymGuard. If `make` is unavailable, run the python commands directly --
# they are listed in README.md.
PY := python
SRC := src

.PHONY: test baseline figures clean all

all: test baseline

test:
	cd $(SRC) && $(PY) -m pytest ../tests -q

baseline:
	cd $(SRC) && $(PY) -m symguard.run_baseline --scenarios 60

# once the public CSV is placed in data/raw/
real:
	cd $(SRC) && $(PY) -m symguard.run_baseline --csv ../data/raw/classData.csv

figures: baseline

clean:
	rm -f reports/results.csv reports/*.png
	find . -name __pycache__ -type d -exec rm -rf {} +
