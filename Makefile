PYTHON ?= .venv/bin/python
NODE ?= node
CPU_PYTHON ?= python3
TENSORSTORE_PATH ?= /tmp/marin-tensorstore-lib

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

.PHONY: weights-consumer-faults
weights-consumer-faults:
	$(CPU_PYTHON) scripts/probe_weights_consumer_faults.py

.PHONY: tensorstore-io
tensorstore-io:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_tensorstore_roundtrip.py

.PHONY: serialize-real-io
serialize-real-io:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_serialize_arrays_real_io.py

.PHONY: restore-real-io
restore-real-io:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_restore_candidate_real_io.py

.PHONY: tree-restore-contracts
tree-restore-contracts:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_tree_restore_contracts.py

.PHONY: grug-state-restore
grug-state-restore:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_grug_state_restore_real_io.py

.PHONY: restore-data-clock
restore-data-clock:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_restore_data_clock.py
	$(PYTHON) scripts/plot_restore_data_clock.py

.PHONY: mixture-boundary-logging
mixture-boundary-logging:
	$(CPU_PYTHON) scripts/probe_mixture_boundary_logging.py
	$(PYTHON) scripts/plot_mixture_boundary_logging.py

.PHONY: loss-composition
loss-composition:
	$(CPU_PYTHON) scripts/probe_loss_composition_cpu.py

.PHONY: loss-cross-replay
loss-cross-replay:
	$(CPU_PYTHON) scripts/probe_loss_cross_replay_cpu.py
	$(PYTHON) scripts/plot_loss_cross_replay.py

.PHONY: tagged-eval-accumulator
tagged-eval-accumulator:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_tagged_eval_accumulator_cpu.py

.PHONY: eval-callback-shapes
eval-callback-shapes:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_eval_callback_shapes_cpu.py

.PHONY: microbatch-loss-cpu
microbatch-loss-cpu:
	PYTHONPATH=$(TENSORSTORE_PATH) $(CPU_PYTHON) scripts/probe_microbatch_loss_cpu.py

.PHONY: engineering-audit-v77
engineering-audit-v77:
	$(PYTHON) scripts/audit_engineering_v77.py

.PHONY: portable-expert-cpu
portable-expert-cpu:
	$(CPU_PYTHON) scripts/probe_portable_expert_mlp_cpu.py
	$(PYTHON) scripts/plot_portable_expert_mlp.py

.PHONY: receiver-layout-cpu
receiver-layout-cpu:
	$(CPU_PYTHON) scripts/probe_receiver_layout_cpu.py
	$(PYTHON) scripts/plot_receiver_layout.py

.PHONY: post-clip-router-cpu
post-clip-router-cpu:
	$(CPU_PYTHON) scripts/probe_post_clip_router_cpu.py
	$(PYTHON) scripts/plot_post_clip_router.py

.PHONY: portable-router-precision-cpu
portable-router-precision-cpu:
	$(CPU_PYTHON) scripts/probe_portable_router_precision_cpu.py

.PHONY: runtime-defaults-probe
runtime-defaults-probe:
	$(PYTHON) scripts/probe_runtime_defaults.py

.PHONY: launch-binding-probe
launch-binding-probe:
	$(PYTHON) scripts/probe_launch_binding.py

.PHONY: clipping-precision-cpu
clipping-precision-cpu:
	$(CPU_PYTHON) scripts/probe_clipping_precision_cpu.py


.PHONY: decay-resume-cpu
decay-resume-cpu:
	$(CPU_PYTHON) scripts/probe_decay_resume_cpu.py


HOST_PYTHON ?= python3
.PHONY: phase-budget-probe
phase-budget-probe:
	$(HOST_PYTHON) scripts/probe_phase_budget.py


.PHONY: integer-exposure-cpu
integer-exposure-cpu:
	$(CPU_PYTHON) scripts/probe_integer_exposure_cpu.py

.PHONY: resume-update-identity
resume-update-identity:
	$(CPU_PYTHON) scripts/probe_resume_update_identity.py

.PHONY: seed-pipeline-cpu
seed-pipeline-cpu:
	$(CPU_PYTHON) scripts/probe_seed_pipeline_cpu.py

.PHONY: loss-mass-cpu
loss-mass-cpu:
	$(CPU_PYTHON) scripts/probe_loss_mass_cpu.py
