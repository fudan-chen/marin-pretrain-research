PYTHON ?= .venv/bin/python
NODE ?= node
CPU_PYTHON ?= python3

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
	$(PYTHON) scripts/probe_eval_format.py
	$(PYTHON) scripts/analyze_live_2026_10_07.py > analysis/live_2026_10_07_computation_log.txt
	$(PYTHON) scripts/write_live_2026_10_07.py
	$(PYTHON) scripts/audit_default_target_weights.py
	$(PYTHON) scripts/write_masked_numerics.py
	$(PYTHON) scripts/write_zero_gradient_state.py
	$(PYTHON) scripts/write_group_clipping.py
	$(PYTHON) scripts/probe_gradient_accumulation.py
	$(PYTHON) scripts/infer_eval_weights.py
	$(PYTHON) scripts/write_eval_weight_inference.py
	$(PYTHON) scripts/test_repeat_exposure.py
	$(PYTHON) scripts/probe_parallel_packing.py
	$(PYTHON) scripts/test_eval_target_alignment.py
	$(PYTHON) scripts/test_eval_array_export.py
	$(PYTHON) scripts/test_eval_replays.py
	$(PYTHON) scripts/probe_eval_identity.py
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

.PHONY: cpu-numerics
cpu-numerics:
	$(CPU_PYTHON) scripts/probe_masked_numerics_cpu.py
	$(PYTHON) scripts/audit_default_target_weights.py
	$(PYTHON) scripts/write_masked_numerics.py

.PHONY: optimizer-cpu
optimizer-cpu:
	$(CPU_PYTHON) scripts/probe_zero_gradient_state_cpu.py
	$(PYTHON) scripts/write_zero_gradient_state.py

.PHONY: clipping-cpu
clipping-cpu:
	$(CPU_PYTHON) scripts/probe_group_clipping_cpu.py
	$(PYTHON) scripts/write_group_clipping.py

.PHONY: loader-resume
loader-resume:
	$(CPU_PYTHON) scripts/probe_loader_resume.py

.PHONY: mixture-identity
mixture-identity:
	$(CPU_PYTHON) scripts/probe_mixture_identity_cpu.py

.PHONY: inner-shuffle
inner-shuffle:
	$(CPU_PYTHON) scripts/probe_inner_shuffle_cpu.py

.PHONY: budget-inventory
budget-inventory:
	$(CPU_PYTHON) scripts/probe_budget_inventory_cpu.py

.PHONY: routing-proposal
routing-proposal:
	$(CPU_PYTHON) scripts/probe_routing_gradient_proposal_cpu.py
	$(PYTHON) scripts/audit_moe_proposal_sources.py

.PHONY: moe-performance
moe-performance:
	$(PYTHON) scripts/analyze_moe_performance.py
	$(PYTHON) scripts/plot_moe_performance.py

.PHONY: routing-envelope
routing-envelope:
	$(CPU_PYTHON) scripts/probe_routing_gradient_envelope_cpu.py
	$(PYTHON) scripts/plot_routing_gradient_envelope.py

.PHONY: router-weight-path
router-weight-path:
	$(CPU_PYTHON) scripts/probe_router_weight_path_cpu.py

.PHONY: router-coupling-update
router-coupling-update:
	$(CPU_PYTHON) scripts/probe_router_coupling_update_cpu.py
	$(PYTHON) scripts/plot_router_coupling_update.py

.PHONY: mix-event-identifiability
mix-event-identifiability:
	$(PYTHON) scripts/analyze_mix_event_identifiability.py
	$(PYTHON) scripts/plot_mix_event_identifiability.py

.PHONY: swarm-seed-pairs
swarm-seed-pairs:
	$(PYTHON) scripts/analyze_swarm_seed_pairs.py
	$(PYTHON) scripts/plot_swarm_seed_pairs.py

.PHONY: pending-router-view
pending-router-view:
	$(CPU_PYTHON) scripts/probe_pending_router_view_cpu.py
