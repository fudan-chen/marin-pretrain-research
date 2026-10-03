PYTHON ?= .venv/bin/python

.PHONY: report serve
report:
	$(PYTHON) scripts/analyze.py > analysis/computation_log.txt
	$(PYTHON) scripts/appendices.py
	$(PYTHON) scripts/charts.py
	$(PYTHON) scripts/build_html.py
	$(PYTHON) scripts/archive_manifest.py
	$(PYTHON) scripts/validate.py

serve:
	$(PYTHON) -m http.server 8765 --bind 127.0.0.1
