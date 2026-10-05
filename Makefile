PYTHON ?= .venv/bin/python
NODE ?= node

.PHONY: report serve
report:
	$(PYTHON) scripts/analyze.py > analysis/computation_log.txt
	$(PYTHON) scripts/appendices.py
	$(PYTHON) scripts/analyze_deepening.py > analysis/deepening_computation_log.txt
	$(PYTHON) scripts/analyze_practical.py > analysis/practical_computation_log.txt
	$(PYTHON) scripts/analyze_findings.py > analysis/findings_computation_log.txt
	$(PYTHON) scripts/analyze_execution.py > analysis/execution_computation_log.txt
	$(PYTHON) scripts/analyze_scale.py > analysis/scale_computation_log.txt
	$(PYTHON) scripts/build_learning.py
	$(PYTHON) scripts/build_decision.py
	$(PYTHON) scripts/build_transfer.py > analysis/transfer_computation_log.txt
	$(PYTHON) scripts/build_order.py
	$(PYTHON) scripts/build_engineering.py
	$(PYTHON) scripts/build_assessment.py
	$(PYTHON) scripts/charts.py
	$(PYTHON) scripts/charts_deepening.py
	$(PYTHON) scripts/charts_practical.py
	$(PYTHON) scripts/charts_findings.py
	$(PYTHON) scripts/charts_transfer.py
	$(PYTHON) scripts/charts_order.py
	$(PYTHON) scripts/charts_scale.py
	$(NODE) scripts/test_planner.cjs
	$(NODE) scripts/test_review.cjs
	$(NODE) scripts/test_decision.cjs
	$(PYTHON) scripts/test_transfer.py
	$(PYTHON) scripts/test_order.py
	$(NODE) scripts/test_order.cjs
	$(NODE) scripts/test_assessment.cjs
	$(PYTHON) scripts/build_html.py
	$(PYTHON) scripts/archive_manifest.py
	$(PYTHON) scripts/validate.py

serve:
	$(PYTHON) -m http.server 8765 --bind 127.0.0.1
