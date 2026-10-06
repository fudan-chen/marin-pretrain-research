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
	$(PYTHON) scripts/probe_contracts.py
	$(PYTHON) scripts/probe_state.py
	$(PYTHON) scripts/probe_boundaries.py
	$(PYTHON) scripts/audit_cache.py
	$(PYTHON) scripts/probe_dedup.py
	$(PYTHON) scripts/probe_routing.py
	$(PYTHON) scripts/probe_qb_partition.py
	$(PYTHON) scripts/probe_optimizer.py
	$(PYTHON) scripts/analyze_live_2026_10_06.py
	$(PYTHON) scripts/probe_mixture_range.py
	$(PYTHON) scripts/probe_batch_clock.py
	$(PYTHON) scripts/probe_failure_loop.py
	$(PYTHON) scripts/probe_watch.py
	$(PYTHON) scripts/test_optimizer_bundle.py
	$(PYTHON) scripts/probe_adamh.py
	$(PYTHON) scripts/draw_adamh_state.py
	$(PYTHON) scripts/probe_muon_direction.py
	$(PYTHON) scripts/draw_muon_direction.py
	$(PYTHON) scripts/probe_muon_geometry.py
	$(PYTHON) scripts/draw_muon_geometry.py
	$(PYTHON) scripts/probe_short_conv.py
	$(PYTHON) scripts/probe_checkpoint_memory.py
	$(PYTHON) scripts/probe_checkpoint_commit.py
	$(PYTHON) scripts/audit_eval_metrics.py
	$(PYTHON) scripts/analyze_mix_trajectory.py > analysis/mix_trajectory_computation.txt
	$(PYTHON) scripts/audit_router_precision.py
	$(PYTHON) scripts/audit_ladder_contract.py
	$(PYTHON) scripts/probe_quality.py
	$(PYTHON) scripts/build_quality_windows.py
	$(NODE) scripts/test_quality.cjs
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
	$(PYTHON) scripts/build_engineering_map.py
	$(PYTHON) scripts/build_html.py
	$(PYTHON) scripts/archive_manifest.py
	$(PYTHON) scripts/validate.py

serve:
	$(PYTHON) -m http.server 8765 --bind 127.0.0.1
