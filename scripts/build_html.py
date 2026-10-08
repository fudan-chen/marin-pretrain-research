# -*- coding: utf-8 -*-
"""Build a file:// compatible report with embedded data and no remote assets."""
import pathlib,json,base64,html,re
import markdown
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).resolve().parents[1]
chapters=[('learning','理解文档与自测','LEARNING_GUIDE_ZH.md'),('rubrics','18条判断规则','RUBRICS_ZH.md'),('pipeline','七阶段决策管线','PIPELINE_ZH.md'),('improvement','持续改进队列','IMPROVEMENT_LOG_ZH.md'),('conclusions','第四轮：结论与判断','CONCLUSIONS_ZH.md'),('report','研究主报告','REPORT_ZH.md'),('data','数据配比与顺序','DATA_GUIDE_ZH.md'),('deep','第二轮：搜索、取舍与游标','DEEP_DIVE_ZH.md'),('practical','第三轮：删域、曝光与顺序','PRACTICAL_ZH.md'),('translation','#8435逐条中文解读','ISSUE_8435_ZH.md'),('operations','运行记录与后续更新','OPERATIONS_ZH.md'),('provenance','来源与复核','PROVENANCE_ZH.md')]
chapters.insert(0,('decision-guide','候选选择与规则回放','DECISION_GUIDE_ZH.md'))
chapters.insert(0,('transfer-guide','预算分解与供体实验','TRANSFER_GUIDE_ZH.md'))
chapters.insert(0,('recent-moe-guide','新MoE提案与数值合同','RECENT_MOE_CHANGES_ZH.md'))
chapters.insert(0,('mixture-identity-guide','混合数据映射身份','MIXTURE_IDENTITY_ZH.md'))
chapters.insert(0,('order-guide','顺序、边缘块与索引','ORDER_GUIDE_ZH.md'))
chapters.insert(0,('engineering-guide','工程排障、反证与验收','ENGINEERING_GUIDE_ZH.md'))
chapters.insert(0,('assessment-guide','理解检查、锚点与分歧','ASSESSMENT_GUIDE_ZH.md'))
chapters.insert(0,('scale-guide','跨规模比较、评估口径与确认','SCALE_TRANSFER_ZH.md'))
chapters.insert(0,('contracts-guide','源码接口、梯度与恢复验收','IMPLEMENTATION_CONTRACTS_ZH.md'))
chapters.insert(0,('state-guide','训练状态、精度与切换时刻','TRAIN_STATE_ZH.md'))
chapters.insert(0,('cache-guide','缓存身份、分词与配比归因','CACHE_PROVENANCE_ZH.md'))
chapters.insert(0,('dedup-guide','去重、样本对齐与曝光解释','DEDUP_FILTERS_ZH.md'))
chapters.insert(0,('quality-guide','质量评分、观察窗口与配比','QUALITY_BUCKETS_ZH.md'))
chapters.insert(0,('change-guide','训练变更评审与执行入口','CHANGE_REVIEW_ZH.md'))
chapters.insert(0,('routing-guide','MoE分配丢弃、权重与目标','ROUTING_DROPS_ZH.md'))
chapters.insert(0,('boundary-guide','文档边界、上下文与有效目标','DOCUMENT_BOUNDARIES_ZH.md'))
chapters.insert(0,('qb-guide','路由均衡、分位数与数据分组','QB_ESTIMATION_ZH.md'))
chapters.insert(0,('optimizer-guide','优化器分组、衰减与状态时钟','OPTIMIZER_GROUPS_ZH.md'))
chapters.insert(0,('loss-triage','配比切换后的loss诊断流程','LOSS_TRIAGE_ZH.md'))
chapters.insert(0,('short-conv-guide','ShortConv文档边界、halo与数值','SHORT_CONV_ZH.md'))
chapters.insert(0,('router-precision-guide','Router精度与训练评估策略','ROUTER_PRECISION_ZH.md'))
chapters.insert(0,('mix-trajectory-guide','配比切换与16域真实轨迹','MIX_TRAJECTORY_ZH.md'))
chapters.insert(0,('eval-metrics-guide','评估字段、父级聚合与分母','EVAL_METRICS_ZH.md'))
chapters.insert(0,('checkpoint-commit-guide','Checkpoint提交、回退与可恢复进度','CHECKPOINT_COMMIT_ZH.md'))
chapters.insert(0,('checkpoint-memory-guide','Checkpoint内存预算、副本与分块','CHECKPOINT_MEMORY_ZH.md'))
chapters.insert(0,('engineering-map-guide','按症状阅读：15案与六条检查入口','ENGINEERING_MAP_ZH.md'))
chapters.insert(0,('delivery-audit-guide','当前交付与真实验证范围','DELIVERY_AUDIT_ZH.md'))
chapters.insert(0,('muon-geometry-guide','MuonH范数、分片与更新几何','MUON_GEOMETRY_ZH.md'))
chapters.insert(0,('adamh-state-guide','AdamH动量、计数与恢复方向','ADAMH_STATE_ZH.md'))
chapters.insert(0,('muon-direction-guide','Muon方向、NS迭代与矩阵布局','MUON_DIRECTION_ZH.md'))
chapters.insert(0,('observability-guide','监控成本、记录时刻与事故重放','OBSERVABILITY_ZH.md'))
chapters.insert(0,('failure-boundaries-guide','有限loss、失败路径与数值验收水位','FAILURE_BOUNDARIES_ZH.md'))
chapters.insert(0,('batch-clock-guide','配比阶段、batch前缀与日志时钟','BATCH_CLOCK_ZH.md'))
chapters.insert(0,('run-code-guide','历史加载器、源码快照与恢复证据','RUN_CODE_PROVENANCE_ZH.md'))
chapters.insert(0,('mixture-range-guide','大曝光量、混合计数与索引范围','MIXTURE_RANGE_ZH.md'))
chapters.insert(0,('live-observation-guide','10月6日：新进度与16域端点观察','LIVE_2026_10_06_ZH.md'))
chapters.insert(0,('eval-identity-guide','评估输入身份、重复遍历与末批','EVAL_IDENTITY_ZH.md'))
chapters.insert(0,('eval-replay-guide','重复评分记录、分母与可执行对照','EVAL_REPLAY_ZH.md'))
chapters.insert(0,('synthesis-guide','结论与操作顺序：怎样用这份研究','SYNTHESIS_ZH.md'))
chapters.insert(0,('eval-array-export-guide','评估数组、输入摘要与记录导出','EVAL_ARRAY_EXPORT_ZH.md'))
chapters.insert(0,('eval-format-guide','Paloma评分格式与连续token流','EVAL_FORMAT_ZH.md'))
chapters.insert(0,('eval-target-alignment-guide','下一token对齐与有效目标覆盖','EVAL_TARGET_ALIGNMENT_ZH.md'))
chapters.insert(0,('packing-fields-guide','打包字段对齐与权重坐标','PACKING_FIELDS_ZH.md'))
chapters.insert(0,('repeat-exposure-guide','重复曝光与配比加量','REPEAT_EXPOSURE_ZH.md'))
chapters.insert(0,('live-oct7-guide','10月7日：macro回升与micro下降','LIVE_2026_10_07_ZH.md'))
chapters.insert(0,('eval-weight-guide','micro权重反推与PTB均值贡献','EVAL_WEIGHT_INFERENCE_ZH.md'))
chapters.insert(0,('gradient-accumulation-guide','梯度累积、有效分母与更新方向','GRADIENT_ACCUMULATION_ZH.md'))
chapters.insert(0,('masked-numerics-guide','零权重、非有限值与反向梯度','MASKED_NUMERICS_ZH.md'))
chapters.insert(0,('zero-gradient-state-guide','零梯度、空目标与训练时钟','ZERO_GRADIENT_STATE_ZH.md'))
chapters.insert(0,('controller-recovery-guide','控制器备份、回滚与失败保护','CONTROLLER_RECOVERY_ZH.md'))
chapters.insert(0,('gang-recovery-guide','Gang重试、预算与资源归还','GANG_RECOVERY_ZH.md'))
chapters.insert(0,('operational-progress-guide','日历进度、步时与事故对齐','OPERATIONAL_PROGRESS_ZH.md'))
chapters.insert(0,('pending-resume-guide','Pending恢复、评估视图与EMA donation','PENDING_RESUME_ZH.md'))
chapters.insert(0,('training-decisions-guide','训练决策：什么时候值得改数据','TRAINING_DECISIONS_ZH.md'))
chapters.insert(0,('experiment-controls-guide','实验对照：权重、子集、顺序与恢复','EXPERIMENT_CONTROLS_ZH.md'))
priority=['synthesis-guide','training-decisions-guide','experiment-controls-guide','pending-resume-guide','operational-progress-guide','gang-recovery-guide','controller-recovery-guide','optimizer-guide','zero-gradient-state-guide','masked-numerics-guide','gradient-accumulation-guide','eval-weight-guide','live-oct7-guide','repeat-exposure-guide','packing-fields-guide','eval-target-alignment-guide','eval-format-guide','eval-array-export-guide','eval-replay-guide','eval-identity-guide','live-observation-guide','mixture-range-guide','batch-clock-guide','failure-boundaries-guide','observability-guide','muon-direction-guide','adamh-state-guide','muon-geometry-guide','engineering-map-guide','delivery-audit-guide','checkpoint-memory-guide','checkpoint-commit-guide','eval-metrics-guide','mix-trajectory-guide','router-precision-guide','short-conv-guide','loss-triage','qb-guide','routing-guide','change-guide','quality-guide','dedup-guide','cache-guide','boundary-guide','state-guide','contracts-guide','scale-guide']
chapters.sort(key=lambda c:priority.index(c[0]) if c[0] in priority else len(priority))
readings={x['file']:x for x in json.loads((ROOT/'analysis/figure_readings.json').read_text())}
sections=[];toc=[]
for slug,label,file in chapters:
    result=markdown.markdown((ROOT/file).read_text(),extensions=['tables','fenced_code','toc'])
    if slug=='practical':
        fragment=BeautifulSoup(result,'html.parser')
        diagram=BeautifulSoup((ROOT/'assets/ablation_context.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        for placeholder in fragment.select('img[src="assets/ablation_context.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='failure-boundaries-guide':
        fragment=BeautifulSoup(result,'html.parser')
        diagram=BeautifulSoup((ROOT/'assets/pending_finite_gap.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        for placeholder in fragment.select('img[src="assets/pending_finite_gap.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='pending-resume-guide':
        fragment=BeautifulSoup(result,'html.parser')
        for filename in ['pending_resume_step.svg','ema_alias_matrix.svg']:
            diagram=BeautifulSoup((ROOT/'assets'/filename).read_text(),'html.parser').svg
            diagram['style']='display:block;width:100%;height:auto'
            for placeholder in fragment.select('img[src="assets/'+filename+'"]'):
                placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='operational-progress-guide':
        diagram=BeautifulSoup((ROOT/'assets/operational_progress.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/operational_progress.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='data':
        diagram=BeautifulSoup((ROOT/'assets/loss_mass_accounting.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/loss_mass_accounting.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='mixture-identity-guide':
        diagram=BeautifulSoup((ROOT/'assets/seed_pipeline_keys.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/seed_pipeline_keys.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='checkpoint-commit-guide':
        diagram=BeautifulSoup((ROOT/'assets/resume_update_identity.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/resume_update_identity.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='checkpoint-memory-guide':
        diagram=BeautifulSoup((ROOT/'assets/donation_snapshot_flow.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/donation_snapshot_flow.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='controller-recovery-guide':
        diagram=BeautifulSoup((ROOT/'assets/controller_restore_contract.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/controller_restore_contract.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='gang-recovery-guide':
        diagram=BeautifulSoup((ROOT/'assets/gang_recovery_flow.svg').read_text(),'html.parser').svg
        diagram['style']='display:block;width:100%;height:auto'
        fragment=BeautifulSoup(result,'html.parser')
        for placeholder in fragment.select('img[src="assets/gang_recovery_flow.svg"]'):
            placeholder.replace_with(diagram)
        result=str(fragment)
    if slug=='recent-moe-guide':
        perf_svg=BeautifulSoup((ROOT/'assets/moe_performance_attribution.svg').read_text(),'html.parser').svg
        perf_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(perf_svg)+'<figcaption>图A为作者冻结单rack测量，图B为端点算术，图C为机制示意；没有本地GPU复现或原始device trace。</figcaption></figure>'
    if slug=='recent-moe-guide':
        env_svg=BeautifulSoup((ROOT/'assets/routing_gradient_envelope.svg').read_text(),'html.parser').svg
        env_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(env_svg)+'<figcaption>48格选定CPU标量输入，非生产发生率；局部dS差异仍需追踪至router梯度、参数更新和固定评估。</figcaption></figure>'
    if slug=='recent-moe-guide':
        coupling_svg=BeautifulSoup((ROOT/'assets/router_coupling_update.svg').read_text(),'html.parser').svg
        coupling_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(coupling_svg)+'<figcaption>人工CPU集成：原router/portable梯度/衰减包装与真实Optax；expert常数、路径与reshard替身。fresh/warm差异非生产风险估计。</figcaption></figure>'
    if slug=='mix-trajectory-guide':
        event_svg=BeautifulSoup((ROOT/'assets/mix_event_identifiability.svg').read_text(),'html.parser').svg
        event_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(event_svg)+'<figcaption>真实归档时刻的事件设计审计；没有估计配比或执行因果效应。补点的秩控制是人工设计，不是新loss记录。</figcaption></figure>'
    if slug=='mix-trajectory-guide':
        pairs_svg=BeautifulSoup((ROOT/'assets/swarm_seed_pairs.svg').read_text(),'html.parser').svg
        pairs_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(pairs_svg)+'<figcaption>d512来源观察逐seed标签差值；没有置信区间。该swarm参与选优，不是独立确认，也不是535B生产收益。</figcaption></figure>'
    if slug=='optimizer-guide':
        clip_svg=BeautifulSoup((ROOT/'assets/group_clipping_cpu.svg').read_text(),'html.parser').svg
        clip_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(clip_svg)+'<figcaption>真实Optax CPU、人工分组标签；最新归档声明裁剪为None，图示不是Hero已启用的配置。</figcaption></figure>'
    if slug=='zero-gradient-state-guide':
        zero_svg=BeautifulSoup((ROOT/'assets/zero_gradient_state_cpu.svg').read_text(),'html.parser').svg
        zero_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(zero_svg)+'<figcaption>原AdamH模块的人工矩阵CPU轨迹；moment与update各按自己的首个零梯度步归一化，未重放Hero训练。</figcaption></figure>'
    if slug=='masked-numerics-guide':
        mask_svg=BeautifulSoup((ROOT/'assets/masked_numerics_cpu.svg').read_text(),'html.parser').svg
        mask_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(mask_svg)+'<figcaption>真实JAX CPU检查：绿色为有限，红色显示非有限梯度元素数；不是Hero批次或GPU/TPU重放。</figcaption></figure>'
    if slug=='gradient-accumulation-guide':
        gradient_svg=BeautifulSoup((ROOT/'assets/gradient_accumulation.svg').read_text(),'html.parser').svg
        gradient_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(gradient_svg)+'<figcaption>同一组加权目标的解析标量梯度；不是模型、JAX或多卡实验。</figcaption></figure>'
    if slug=='eval-weight-guide':
        weights_svg=BeautifulSoup((ROOT/'assets/eval_weight_inference.svg').read_text(),'html.parser').svg
        weights_svg['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(weights_svg)+'<figcaption>固定权重假设下的条件反推；不是实际分母，也不是训练配比。</figcaption></figure>'
    if slug=='live-oct7-guide':
        october=BeautifulSoup((ROOT/'assets/live_2026_10_07.svg').read_text(),'html.parser').svg
        october['style']='display:block;width:100%;height:auto'
        result += '<figure class="scientific-figure">'+str(october)+'<figcaption>209999→212999的实测端点差。红为上升，蓝为下降；不是配比因果效应。</figcaption></figure>'
    soup=BeautifulSoup(result,'html.parser')
    routes_placeholder=soup.select_one('#diagnostic-routes-placeholder')
    if routes_placeholder is not None:
        route_data=json.loads((ROOT/'analysis/diagnostic_routes.json').read_text())
        route_parts=['<div class="diagnostic-routes" id="diagnostic-routes">']
        for route in route_data['routes']:
            route_parts.append('<details id="diagnostic-route-'+html.escape(route['id'],quote=True)+'"><summary>'+html.escape(route['symptom'])+'</summary><div class="route-body">')
            for label,key in [('先查什么','first'),('源码说明','mechanism'),('最小对照','control'),('怎样决定下一步','decision')]:
                route_parts.append('<p><strong>'+label+'：</strong>'+html.escape(route[key])+'</p>')
            route_parts.append('<p class="route-evidence"><a href="'+html.escape(route['chapter'],quote=True)+'">进入完整解释</a>')
            for e in route['evidence']:
                route_parts.append('<a href="'+html.escape(e['file'],quote=True)+'">结果记录（'+str(e['checks_passed'])+'项局部检查）</a>')
            route_parts.append('</p><p class="route-boundary"><strong>证据范围：</strong>'+html.escape(route['limit'])+'</p></div></details>')
        route_parts.append('</div>')
        routes_placeholder.replace_with(BeautifulSoup(''.join(route_parts),'html.parser'))
    stall_placeholder=soup.select_one('#loader-stall-flow-placeholder')
    if stall_placeholder is not None:
        stall_svg=BeautifulSoup((ROOT/'assets/loader_stall_flow.svg').read_text(),'html.parser').svg
        stall_svg['style']='display:block;width:100%;min-width:800px;height:auto;'
        stall_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        stall_wrap.append(stall_svg);stall_placeholder.replace_with(stall_wrap)
    live_placeholder=soup.select_one('#live-observation-placeholder')
    if live_placeholder is not None:
        live_svg=BeautifulSoup((ROOT/'assets/live_2026_10_06.svg').read_text(),'html.parser').svg
        live_svg['style']='display:block;width:100%;min-width:800px;height:auto;'
        live_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        live_wrap.append(live_svg);live_placeholder.replace_with(live_wrap)
    direction_placeholder=soup.select_one('#muon-direction-placeholder')
    if direction_placeholder is not None:
        direction_svg=BeautifulSoup((ROOT/'assets/muon_direction.svg').read_text(),'html.parser').svg
        direction_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        direction_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        direction_wrap.append(direction_svg);direction_placeholder.replace_with(direction_wrap)
    adamh_placeholder=soup.select_one('#adamh-state-placeholder')
    if adamh_placeholder is not None:
        adamh_svg=BeautifulSoup((ROOT/'assets/adamh_state.svg').read_text(),'html.parser').svg
        adamh_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        adamh_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        adamh_wrap.append(adamh_svg);adamh_placeholder.replace_with(adamh_wrap)
    muon_placeholder=soup.select_one('#muon-geometry-placeholder')
    if muon_placeholder is not None:
        muon_svg=BeautifulSoup((ROOT/'assets/muon_geometry.svg').read_text(),'html.parser').svg
        muon_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        muon_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        muon_wrap.append(muon_svg);muon_placeholder.replace_with(muon_wrap)
    postclip_placeholder=soup.select_one('#post-clip-router-placeholder')
    if postclip_placeholder is not None:
        postclip_svg=BeautifulSoup((ROOT/'assets/post_clip_router.svg').read_text(),'html.parser').svg
        postclip_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        postclip_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        postclip_wrap.append(postclip_svg);postclip_placeholder.replace_with(postclip_wrap)
    receiver_placeholder=soup.select_one('#receiver-layout-placeholder')
    if receiver_placeholder is not None:
        receiver_svg=BeautifulSoup((ROOT/'assets/receiver_layout.svg').read_text(),'html.parser').svg
        receiver_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        receiver_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        receiver_wrap.append(receiver_svg);receiver_placeholder.replace_with(receiver_wrap)
    portable_placeholder=soup.select_one('#portable-expert-placeholder')
    if portable_placeholder is not None:
        portable_svg=BeautifulSoup((ROOT/'assets/portable_expert_mlp.svg').read_text(),'html.parser').svg
        portable_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        portable_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        portable_wrap.append(portable_svg);portable_placeholder.replace_with(portable_wrap)
    cross_placeholder=soup.select_one('#loss-cross-replay-placeholder')
    if cross_placeholder is not None:
        cross_svg=BeautifulSoup((ROOT/'assets/loss_cross_replay.svg').read_text(),'html.parser').svg
        cross_svg['style']='display:block;width:100%;min-width:700px;height:auto;'
        cross_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        cross_wrap.append(cross_svg);cross_placeholder.replace_with(cross_wrap)
    boundary_placeholder=soup.select_one('#mixture-boundary-log-placeholder')
    if boundary_placeholder is not None:
        boundary_svg=BeautifulSoup((ROOT/'assets/mixture_boundary_logging.svg').read_text(),'html.parser').svg
        boundary_svg['style']='display:block;width:100%;height:auto;'
        boundary_placeholder.replace_with(boundary_svg)
    resume_placeholder=soup.select_one('#ablation-resume-boundary-placeholder')
    if resume_placeholder is not None:
        resume_svg=BeautifulSoup((ROOT/'assets/ablation_resume_boundary.svg').read_text(),'html.parser').svg
        resume_svg['style']='display:block;width:100%;height:auto;'
        resume_placeholder.replace_with(resume_svg)
    prefix_placeholder=soup.select_one('#prefix-sampling-placeholder')
    if prefix_placeholder is not None:
        prefix_svg=BeautifulSoup((ROOT/'assets/prefix_sampling_structure.svg').read_text(),'html.parser').svg
        prefix_svg['style']='display:block;width:100%;height:auto;'
        prefix_placeholder.replace_with(prefix_svg)
    sim_placeholder=soup.select_one('#simulated-inventory-placeholder')
    if sim_placeholder is not None:
        sim_svg=BeautifulSoup((ROOT/'assets/simulated_inventory.svg').read_text(),'html.parser').svg
        sim_svg['style']='display:block;width:100%;min-width:760px;height:auto;'
        sim_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        sim_wrap.append(sim_svg);sim_placeholder.replace_with(sim_wrap)
    key_placeholder=soup.select_one('#historical-key-evidence-placeholder')
    if key_placeholder is not None:
        key_svg=BeautifulSoup((ROOT/'assets/historical_key_evidence.svg').read_text(),'html.parser').svg
        key_svg['style']='display:block;width:100%;min-width:760px;height:auto;'
        key_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        key_wrap.append(key_svg);key_placeholder.replace_with(key_wrap)
    code_placeholder=soup.select_one('#run-code-evidence-placeholder')
    if code_placeholder is not None:
        code_svg=BeautifulSoup((ROOT/'assets/run_code_evidence.svg').read_text(),'html.parser').svg
        code_svg['style']='display:block;width:100%;min-width:760px;height:auto;'
        code_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        code_wrap.append(code_svg);code_placeholder.replace_with(code_wrap)
    clock_placeholder=soup.select_one('#restore-data-clock-placeholder')
    if clock_placeholder is not None:
        clock_svg=BeautifulSoup((ROOT/'assets/restore_data_clock.svg').read_text(),'html.parser').svg
        clock_svg['style']='display:block;width:100%;height:auto;'
        clock_placeholder.replace_with(clock_svg)
    ts_placeholder=soup.select_one('#tensorstore-io-placeholder')
    if ts_placeholder is not None:
        ts_svg=BeautifulSoup((ROOT/'assets/tensorstore_roundtrip.svg').read_text(),'html.parser').svg
        ts_svg['style']='display:block;width:100%;height:auto;'
        ts_placeholder.replace_with(ts_svg)
    checkpoint_placeholder=soup.select_one('#checkpoint-flow-placeholder')
    if checkpoint_placeholder is not None:
        checkpoint_svg=BeautifulSoup((ROOT/'assets/checkpoint_commit_flow.svg').read_text(),'html.parser').svg
        checkpoint_svg['style']='display:block;width:100%;min-width:850px;height:auto;'
        checkpoint_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        checkpoint_wrap.append(checkpoint_svg);checkpoint_placeholder.replace_with(checkpoint_wrap)
    mix_placeholder=soup.select_one('#mix-trajectory-placeholder')
    if mix_placeholder is not None:
        mix_svg=BeautifulSoup((ROOT/'assets/mix_trajectory.svg').read_text(),'html.parser').svg
        mix_svg['style']='display:block;width:100%;min-width:850px;height:auto;'
        mix_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        mix_wrap.append(mix_svg);mix_placeholder.replace_with(mix_wrap)
    drop_placeholder=soup.select_one('#drop-pattern-placeholder')
    if drop_placeholder is not None:
        drop_svg=BeautifulSoup((ROOT/'assets/drop_patterns.svg').read_text(),'html.parser').svg
        drop_svg['style']='display:block;width:100%;min-width:760px;max-width:800px;height:auto;'
        drop_wrap=soup.new_tag('div',attrs={'class':'table-wrap','style':'overflow-x:auto;'})
        drop_wrap.append(drop_svg)
        drop_placeholder.replace_with(drop_wrap)
    placeholder=soup.select_one('#quality-window-placeholder')
    if placeholder is not None:
        frame=soup.new_tag('iframe',attrs={'id':'quality-window-frame','title':'BME字符窗口交互示意','srcdoc':(ROOT/'QUALITY_WINDOWS.html').read_text(),'style':'width:100%;height:780px;border:1px solid #cedde6;border-radius:6px;','loading':'lazy'})
        placeholder.replace_with(frame)
    for h in soup.find_all(re.compile('^h[1-6]$')):
        old=h.get('id','heading');h['id']=slug+'-'+old
        if h.name in ['h2','h3']:toc.append((slug,label,h['id'],h.text,h.name))
    for table in soup.select('table'):
        wrap=soup.new_tag('div',attrs={'class':'table-wrap'});table.wrap(wrap)
        row_count=len(table.select('tbody tr'))
        if row_count>30:
            d=soup.new_tag('details',attrs={'class':'large-table-details'});wrap.wrap(d)
            summary=soup.new_tag('summary');summary.string='展开完整表：%d行，原值与来源保留'%row_count;d.insert(0,summary)
    for fig in soup.select('img[src]'):
        note=readings.get(fig['src'])
        if not note:continue
        fig['id']='figure-'+pathlib.Path(fig['src']).stem
        box=soup.new_tag('div',attrs={'class':'figure-reading'})
        for label,key in [('看什么','question'),('怎么读','reading'),('边界','boundary')]:
            p=soup.new_tag('p');strong=soup.new_tag('strong');strong.string=label+'：';p.append(strong);p.append(note[key]);box.append(p)
        p=soup.new_tag('p')
        for label,key in [('原值/计算入口','values'),('完整解释','chapter')]:
            a=soup.new_tag('a',href=note[key]);a.string=label;p.append(a)
        box.append(p);fig.parent.insert_after(box)
    for a in soup.select('a[href]'):
        if a['href'].startswith('http'):a['target']='_blank';a['rel']='noopener'
    sections.append('<section class="chapter" id="%s">%s</section>'%(slug,str(soup)))
cells=json.loads((ROOT/'analysis/cell_weights.json').read_text());domains={d['id']:d['name_zh'] for d in json.loads((ROOT/'analysis/domain_weights_zh.json').read_text())}
for c in cells:c['domain_zh']=domains[c['domain_id']]
samples=[{'cell':s['cell'],'examples':[t[:300]+('…' if len(t)>300 else '') for t in s['examples']]} for s in json.loads((ROOT/'analysis/sample_texts.json').read_text())]
def data(j):return json.dumps(j,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
nav='<a class="major" href="#workbench">理解与评审工作台</a>'+''.join('<a class="major" href="#%s">%s</a>'%(s,html.escape(l)) for s,l,_ in chapters)
nav+='<a class="major" href="#planner">自己的配比预算工具</a><a class="major" href="#explorer">200桶查询</a><a class="major" href="#notes">我的研究笔记</a>'
nav='<a class="major" href="#decision-lab">候选决策实验室</a>'+nav
nav='<a class="major" href="#transfer-lab">预算分解与交换草案</a>'+nav
nav='<a class="major" href="#order-lab">顺序整数账本</a>'+nav
nav='<a class="major" href="#engineering-lab">工程证据链与下一项检查</a><a class="major" href="#assessment-lab">理解检查与回答记录</a>'+nav
details=''
for slug,label,_ in chapters:
    links=''.join('<a class="%s" href="#%s">%s</a>'%('minor' if kind=='h3' else '',ident,html.escape(text)) for s,_,ident,text,kind in toc if s==slug)
    details+='<details%s><summary>%s · 章节</summary>%s</details>'%(' open' if slug=='learning' else '',html.escape(label),links)
explorer='''<section class="chapter" id="explorer"><h1>200桶查询：比例、库存与计划曝光</h1><p>点击桶号查看公开样本片段。库存与计划抽样量使用不同分母；Cooldown尚未完成。</p><div class="filters"><label>搜索 <input id="bucket-search" placeholder="如 c13q4 / Agent / 数学"></label><label>语义域 <select id="bucket-domain"></select></label><label>质量档 <select id="bucket-quality"><option value="">全部Q档</option><option value="0">Q0</option><option value="1">Q1</option><option value="2">Q2</option><option value="3">Q3</option><option value="4">Q4</option></select></label><label>排序 <select id="bucket-sort"><option value="id">按桶号</option><option value="epochs">按曝光轮数降序</option><option value="delta">按主阶段增幅降序</option></select></label></div><p class="status" id="bucket-count" aria-live="polite"></p><div class="table-wrap bucket-scroll"><table><thead><tr><th>桶 / 样本</th><th>语义域</th><th>库存/B</th><th>原配比</th><th>新主阶段</th><th>Cooldown计划</th><th>全程计划/B</th><th>计划轮数</th></tr></thead><tbody id="bucket-rows"></tbody></table></div><div class="sample-panel" id="sample-panel" hidden></div><p><a href="analysis/cell_weights.csv" download>下载完整200桶CSV</a> · <a href="analysis/domain_weights.csv" download>下载40域CSV</a> · <a href="analysis/summary.json">查看计算摘要</a></p></section>'''
notes='''<section class="chapter notes" id="notes"><h1>我的研究笔记</h1><p class="status">内容仅保存在当前浏览器的本地存储，可导出为文本。更换浏览器、清除网站数据或改用独立HTML时可能需要重新导入。</p><textarea id="local-notes" aria-label="我的研究笔记" placeholder="记录你想验证的假设、具体证据、对自己项目的调整……"></textarea><p id="notes-status" class="status" aria-live="polite"></p><button id="export-notes" class="action">导出笔记</button></section>'''
template='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Marin 535B预训练研究 · 中文报告</title><style>CSS</style></head><body id="top"><div class="layout"><aside class="sidebar"><div class="brand">Marin 535B<br>预训练研究</div><div class="meta">公开快照：2026-10-04<br>重点：故障机制、Loss、数据顺序</div><input id="toc-search" aria-label="筛选章节目录" placeholder="筛选章节标题"><nav>NAV</nav>DETAILS</aside><main><header class="intro"><h1>从一次大规模预训练里，学会定位问题与调整数据</h1><p>逐条读工程记录，把改动、症状、原因和验证对应起来；重新计算配比和曲线，明确哪里仍缺少因果证据。</p><div class="facts"><span>主帖 + 27条评论</span><span>58条生产事件</span><span>40域 × 5质量档</span><span>7段生产run</span><span>934条配比观测</span></div><div class="actions"><a href="#report">从三个案例开始</a><a href="#data">直接看数据分析</a><a href="#deep">第二轮新增结论</a><a href="REPORT_ZH.md">Markdown主报告</a><a href="report_standalone.html" download>下载单文件HTML</a><button class="action" id="print-report">打印 / 保存PDF</button></div></header>SECTIONS EXPLORER NOTES<div class="footer">文字、图与查询器可离线阅读。原网页与来源链接需要网络。每个数值对应归档文件；本机未执行535B训练。</div></main></div><a class="top-link" href="#top">回到目录</a><script type="application/json" id="bucket-data">CELLS</script><script type="application/json" id="sample-data">SAMPLES</script><script>JS</script></body></html>'''
template=template.replace('SECTIONS EXPLORER NOTES','SECTIONS PLANNER EXPLORER NOTES')
template=template.replace('<script>JS</script>','<script type="application/json" id="planner-example">PLANEXAMPLE</script><script>PLANNERCORE</script><script>JS</script><script>PLANNERUI</script>')
template=template.replace('<a href="#deep">第二轮新增结论</a>','<a href="#deep">第二轮新增结论</a><a href="#practical">第三轮：如何设计自己的实验</a><a href="#planner">输入自己的预算</a>')
template=template.replace('<span>934条配比观测</span>','<span>934条配比观测</span><span>129个运行配置复核</span>')
template=template.replace('<a href="#report">从三个案例开始</a>','<a href="#conclusions">先读最新结论与判断</a><a href="#report">从三个案例开始</a>')
template=template.replace('SECTIONS PLANNER','WORKBENCH DECISIONLAB TRANSFERLAB ORDERLAB SECTIONS PLANNER')
template=template.replace('<a href="#conclusions">先读最新结论与判断</a>','<a href="#workbench">从理解与评审工作台开始</a><a href="#learning">理解文档与自测</a><a href="#pipeline">决策管线</a><a href="#conclusions">最新研究结论</a>')
template=template.replace('<script>PLANNERUI</script>','<script>PLANNERUI</script><script type="application/json" id="workbench-data">WORKBENCHDATA</script><script type="application/json" id="rubric-data">RUBRICDATA</script><script>REVIEWCORE</script><script>WORKBENCHJS</script>')
template=template.replace('<script>WORKBENCHJS</script>','<script>WORKBENCHJS</script><script type="application/json" id="decision-data">DECISIONDATA</script><script>DECISIONCORE</script><script>DECISIONUI</script>')
template=template.replace('<a href="#workbench">从理解与评审工作台开始</a>','<a href="#decision-lab">比较候选与退步限制</a><a href="#workbench">从理解与评审工作台开始</a>')
template=template.replace('<a href="#decision-lab">比较候选与退步限制</a>','<a href="#transfer-lab">拆预算并设计供体实验</a><a href="#decision-lab">比较候选与退步限制</a>')
template=template.replace('<script>DECISIONUI</script>','<script>DECISIONUI</script><script type="application/json" id="transfer-data">TRANSFERDATA</script><script>TRANSFERUI</script>')
template=template.replace('<script>TRANSFERUI</script>','<script>TRANSFERUI</script><script type="application/json" id="order-data">ORDERDATA</script><script>ORDERCORE</script><script>ORDERUI</script>')
template=template.replace('<a href="#transfer-lab">拆预算并设计供体实验</a>','<a href="#order-lab">检查顺序与累计量</a><a href="#transfer-lab">拆预算并设计供体实验</a>')
template=template.replace('WORKBENCH DECISIONLAB','ENGINEERINGLAB ASSESSMENTLAB WORKBENCH DECISIONLAB')
template=template.replace('<script>ORDERUI</script>','<script>ORDERUI</script><script type="application/json" id="assessment-data">ASSESSMENTDATA</script><script>ASSESSMENTCORE</script><script>ASSESSMENTUI</script>')
template=template.replace('<a href="#order-lab">检查顺序与累计量</a>','<a href="#assessment-lab">保存自己的证据判断</a><a href="#order-lab">检查顺序与累计量</a>')
template=template.replace('<script>ASSESSMENTUI</script>','<script>ASSESSMENTUI</script><script type="application/json" id="engineering-data">ENGINEERINGDATA</script><script>ENGINEERINGUI</script>')
template=template.replace('公开快照：2026-10-04','最新观察：2026-10-07 · 旧图快照：10-04')
template=template.replace('<a href="#assessment-lab">保存自己的证据判断</a>','<a href="#engineering-lab">核对工程解释与反证</a><a href="#assessment-lab">保存自己的证据判断</a>')
template=template.replace('<a href="#engineering-lab">核对工程解释与反证</a>', '<a href="#engineering-map-guide">从症状选择检查入口</a><a href="#delivery-audit-guide">当前交付与证据边界</a><a href="#checkpoint-memory-guide">保存内存与写入分摊</a><a href="#checkpoint-commit-guide">保存提交与可恢复进度</a><a href="#eval-metrics-guide">评估指标与V25更正</a><a href="#mix-trajectory-guide">配比切换与16域曲线</a><a href="#router-precision-guide">Router精度与评估策略</a><a href="#short-conv-guide">ShortConv边界与halo</a><a href="#loss-triage">loss变化诊断流程</a><a href="#optimizer-guide">优化器分组与衰减</a><a href="#qb-guide">路由均衡与数据分组</a><a href="#routing-guide">MoE丢弃与训练目标</a><a href="#change-guide">训练变更评审</a><a href="#quality-guide">评分窗口与质量桶</a><a href="#dedup-guide">去重与样本对齐</a><a href="#cache-guide">缓存身份与配比归因</a><a href="#boundary-guide">文档边界与有效目标</a><a href="#state-guide">训练状态与切换时刻</a><a href="#contracts-guide">源码接口与恢复验收</a><a href="#scale-guide">配比历史与BPB聚合检查</a><a href="#engineering-lab">核对工程解释与反证</a>')
workbench=json.loads((ROOT/'analysis/workbench_data.json').read_text());rubric=json.loads((ROOT/'analysis/rubrics.json').read_text());chapter_paths={file:'#'+slug for slug,_,file in chapters}
for c in workbench['cases']:c['source']=chapter_paths.get(c['source'],c['source'])
for rule in rubric['rules']:
    for link in rule['evidence_links']:link['href']=chapter_paths.get(link['href'],link['href'])
mapping={'CSS':(ROOT/'assets/report.css').read_text()+'\n'+(ROOT/'assets/workbench.css').read_text(),'NAV':nav,'DETAILS':details,'SECTIONS':'\n'.join(sections),'EXPLORER':explorer,'NOTES':notes,'CELLS':data(cells),'SAMPLES':data(samples),'JS':(ROOT/'assets/report.js').read_text(),
         'WORKBENCH':(ROOT/'assets/workbench.html').read_text(),'WORKBENCHDATA':data(workbench),'RUBRICDATA':data(rubric),'REVIEWCORE':(ROOT/'assets/review-core.js').read_text(),'WORKBENCHJS':(ROOT/'assets/workbench.js').read_text(),
         'PLANNER':(ROOT/'assets/planner.html').read_text(),'PLANEXAMPLE':data(json.loads((ROOT/'analysis/planner_example.json').read_text())),
         'PLANNERCORE':(ROOT/'assets/planner-core.js').read_text(),'PLANNERUI':(ROOT/'assets/planner-ui.js').read_text()}
mapping['CSS']+='\n'+(ROOT/'assets/diagnostic-routes.css').read_text()
mapping['CSS']+='\n'+(ROOT/'assets/decision.css').read_text()
mapping.update({'DECISIONLAB':(ROOT/'assets/decision.html').read_text(),'DECISIONDATA':data(json.loads((ROOT/'analysis/decision_data.json').read_text())),
                'DECISIONCORE':(ROOT/'assets/decision-core.js').read_text(),'DECISIONUI':(ROOT/'assets/decision-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/transfer.css').read_text()
mapping.update({'TRANSFERLAB':(ROOT/'assets/transfer.html').read_text(),'TRANSFERDATA':data(json.loads((ROOT/'analysis/transfer_data.json').read_text())),
                'TRANSFERUI':(ROOT/'assets/transfer-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/order.css').read_text()
mapping.update({'ORDERLAB':(ROOT/'assets/order.html').read_text(),'ORDERDATA':data(json.loads((ROOT/'analysis/order_workbench_data.json').read_text())),
                'ORDERCORE':(ROOT/'assets/order-core.js').read_text(),'ORDERUI':(ROOT/'assets/order-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/assessment.css').read_text()
mapping.update({'ASSESSMENTLAB':(ROOT/'assets/assessment.html').read_text(),'ASSESSMENTDATA':data(json.loads((ROOT/'analysis/assessment_data.json').read_text())),
                'ASSESSMENTCORE':(ROOT/'assets/assessment-core.js').read_text(),'ASSESSMENTUI':(ROOT/'assets/assessment-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/engineering.css').read_text()
mapping.update({'ENGINEERINGLAB':(ROOT/'assets/engineering.html').read_text(),'ENGINEERINGDATA':data(json.loads((ROOT/'analysis/engineering_data.json').read_text())),'ENGINEERINGUI':(ROOT/'assets/engineering-ui.js').read_text()})
# Show the synthesis before interactive labs; retain their ids and behavior.
intro=sections.pop(0)
assert 'id="synthesis-guide"' in intro
decision_intro=sections.pop(0)
assert 'id="training-decisions-guide"' in decision_intro
mapping['SYNTHESISINTRO']=intro+'\n'+decision_intro
mapping['SECTIONS']='\n'.join(sections)
intro_nav='<a class="major" href="#synthesis-guide">结论与操作顺序：怎样用这份研究</a>'
mapping['NAV']=intro_nav+mapping['NAV'].replace(intro_nav,'')
template=template.replace('ENGINEERINGLAB ASSESSMENTLAB WORKBENCH','SYNTHESISINTRO ENGINEERINGLAB ASSESSMENTLAB WORKBENCH')
template=template.replace('<a href="#engineering-lab">核对工程解释与反证</a>','<a href="#synthesis-guide">先读结论与操作顺序</a><a href="#engineering-lab">核对工程解释与反证</a>')
template=template.replace('<a href="#conclusions">最新研究结论</a>','<a href="#conclusions">历史候选结论</a>')
page=re.sub('|'.join(sorted(mapping,key=len,reverse=True)),lambda m:mapping[m[0]],template)
# Link existing markdown artifacts to their corresponding chapters in this combined report.
for slug,_,file in chapters:page=page.replace('href="'+file+'"','href="#'+slug+'"')
# Retain a downloadable markdown link in the header.
page=page.replace('<a href="#report">Markdown主报告</a>','<a href="REPORT_ZH.md" download>Markdown主报告</a>')
page='\n'.join(line.rstrip() for line in page.splitlines())+'\n'
(ROOT/'index.html').write_text(page)
standalone=page
for p in (ROOT/'assets').glob('*.png'):
    standalone=standalone.replace('src="assets/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
(ROOT/'report_standalone.html').write_text(standalone)
print('HTML built:',len(page.encode()),'bytes; standalone:',len(standalone.encode()),'bytes')
