# V122：没有 Git commit，还能怎样核对历史加载器

这轮取得了共同 checkpoint 的生产 run，以及 `proportional-zero-c00-seed0` 续训 run 上传的代码快照。两份清单中的五个相关文件摘要一致；下载后也与清单内容摘要相符。续训日志明确记录了读取 `step-393`、TensorStore 恢复统计，以及随后出现的 `394it` 训练进度。因此，V121 的恢复边界不再只有配置路径和终点步数支持，还多了一个续训 run 的读取日志。

**但目前仍不能把“条件复算中出现逻辑身份重叠”写成“历史训练重复了这些 token”。** 上传的源码、实际导入的模块、恢复后的状态值、缓存中的样本，是不同的证据。这里补上前两者之间的一部分连接，没有读到真实 token 或 checkpoint 内容。

<div id="run-code-evidence-placeholder"></div>

## 取得了什么，分别能证明什么

两个 run 的匿名 GraphQL 查询都成功，`commit` 都是 `null`。每个文件目录只有一页，均为10项，没有 `wandb-metadata.json` 或 `diff.patch` 文件项。这个结果只描述本次取得的目录，不能推断这些文件从未生成，更不能把空 commit 当成未做版本管理。

`_wandb.code_path` 指向 `source-marin_moe-_callable_runner.py`。它的代码清单并不只有这个启动脚本：生产清单含29,890项，续训清单含29,896项，包括仓库源码和大量虚拟环境文件。本轮没有下载整套依赖，只取分析相关的五个源文件。W&B 的文件取得方式和摘要解释参照其[公开 SDK 存储策略实现](https://github.com/wandb/wandb/blob/main/wandb/sdk/artifacts/storage_policies/wandb_storage_policy.py)。下载以清单的文件大小、Base64 MD5 核对，归档再记录 SHA-256；MD5 在这里用来匹配平台清单，不用作独立的安全证明。

|下载文件|与此前探针输入的关系|本轮能支持的判断|
|---|---|---|
|`data/mixture.py`|文件不同；AST 差异仅在 `_get_stage_for_block`|必须用快照再算，不能拿文件名相同当实现相同|
|`schedule.py`|完整文件字节一致|此前 batch 前缀计算所用源码与这两份快照一致|
|`data/text/datasets.py`|完整文件不同；四个选定的支持过滤、构建、child key 与评估命名方法 AST 一致|V120 中历史正权重保留数据支持的源码机制得到更直接的版本支持；没有重跑整个 token 构建过程|
|`data/loader.py`|完整文件不同；此前探针使用的三个异步读取方法 AST 一致|只能确认这三个方法，不能推广为加载器全部行为一致|
|`main/train_lm.py`|两份清单摘要一致，已下载|仅保留为候选入口材料；未确认这个 run 执行的就是该入口|

[查询、源码与日志目录](sources/run_code_2026_10_08/producer_inventory.json)、[生产代码清单](sources/run_code_2026_10_08/producer_code_manifest.json)、[续训代码清单](sources/run_code_2026_10_08/continuation_code_manifest.json)、[逐文件绑定与日志行号](analysis/run_code_provenance.json)。清单约7.9MB一份，离线浏览应优先打开较小的绑定结果。

## 那一处源码变化，会不会推翻上轮结果

此前版本先把阶段起点除以 `block_size`，再用 block 序号搜索所属阶段；上传快照则把当前 block 序号乘回序列坐标，再搜索阶段起点。对于这里合法、对齐到整块的阶段声明，两种表示在对应边界一致。这个判断没有靠文字猜测：本轮用上传的完整混合器类、原批量调度器和同一套有限身份子数据集，重新执行了80组条件复算。

结果是：12项原控制通过，80组结果、16,000行逐桶前缀计数、下一批身份摘要及备用给定 key 的控制，与V121全部一致。主给定 key 下的部分块身份重叠范围仍为5129–5977；它仍然是合成 `(桶, 逻辑索引)` 的集合交，不是实际重复 token 数量。新结果另存于[历史快照复算](analysis/historical_ablation_resume_boundary.json)，没有覆盖V121原始结果。对齐边界的16个方法比较点也一致；这不是对所有整数范围与所有无效配置的证明。

同样，四个数据集方法 AST 一致只支持“相同声明、相同输入下，所审计方法没有版本差异”。本轮没有恢复历史 child shuffle key，也没有新跑80份真实缓存读取。200桶、有限身份数据集、给定 key 等替代条件仍要随结果一起保留。

## 恢复日志比配置多告诉了什么

[生产日志](sources/run_code_2026_10_08/producer_output.log.txt)记录 checkpoint 未找到后从头开始。[续训日志](sources/run_code_2026_10_08/continuation_output.log.txt)记录以下顺序，原始行号在分析JSON内：

1. 从共同生产 run 的 `checkpoints/step-393` 路径加载 checkpoint。
2. TensorStore 报告读取约0.65GiB、物化约1.54GiB，涉及130个数组。
3. 后续训练进度出现 `394it/3.93kit`。

这些是归档日志中的报告，支持这个 run 确实走到了恢复读取及后续训练。`394it` 是进度显示，不能独自当作某个 optimizer、data cursor 或保存 marker 的精确定义；字节量和数组数也不能证明所有叶子语义正确。checkpoint 状态树、真正的下一批输入和历史导入路径仍未取得。更不能用一个续训日志替80个 run 下结论。

## 对自己训练最有用的规则

**改配比时，保存“从哪段历史继续”，比只保存“现在的权重”更有用。** V120发现过去阶段会影响数据支持与 child key 分配；V121发现阶段边界与恢复点之间还有部分混合块。V122进一步说明，复查这些行为时，不能只寻找一个 Git SHA：若 run 上传了源码，应先核对实际使用函数对应的文件，再核对入口和状态。

评审可以按以下顺序推进，每步都要写出实际拿到的材料：

1. 固定生产与续训 run 身份，归档完整文件目录和阶段声明。查不到 commit 就记为空，继续查代码 artifact。
2. 下载关键文件，核对清单摘要；比较具体使用的方法。文件不同不自动判行为不同，局部 AST 相同不自动判整套程序相同。
3. 查恢复选择、读取完成及后续进度日志。把“声明要读”“日志报告已读”“独立检查内容”分开。
4. 在实际源码下复算完整历史前缀及恢复处的部分块。记录 key 来源、缓存身份、子数据集映射和未执行的替代依赖。
5. 进入真实训练确认时，保存导入模块路径与文件摘要、恢复状态步数、混合块位置，以及恢复前后有明确采样坐标的输入身份摘要。只保存总 count 不足以恢复部分乱序块。

这条管线的作用是减少错误归因：加载器历史还没核对清楚时，不应急着把 loss 改善归功于某个数据域，也不应因为合成身份有交集就废弃一套配比。真实训练效应仍要靠输入可比、供体明确的对照与独立确认。

可直接填写[运行谱系检查模板](templates/run_lineage_review.json)。模板为空、状态为未执行；八项本地证据控制通过，不代表这份模板已在生产使用。

## 怎样复查本轮

来源已随报告归档，普通离线阅读不需要重新抓取。`scripts/acquire_run_code_provenance.py` 在已有文件上只核对归档摘要；不存在时才做匿名只读下载。`scripts/analyze_run_code_provenance.py` 使用前轮已明确列出的身份子数据集与局部依赖替代，切换到本轮下载源码后重跑边界控制。

```sh
make run-code-snapshot
make run-code-replay CPU_PYTHON=/path/to/python-with-jax
make run-code-static
```

复算需要Python 3.10以上及可用的JAX CPU环境。本轮实测JAX 0.7.2、NumPy 2.5.3；离线HTML构建和静态检查使用报告已有的Markdown/BeautifulSoup环境。复算不启动GPU训练、不读取真实模型checkpoint。`run-code-static` 检查已经生成的HTML；修改正文后先运行 `scripts/build_html.py`。

## V123：seed相同，还要问哪一条随机流相同

本轮从同一代码artifact再取得三个训练入口、checkpoint辅助代码、`key_iterator`和JAX版本文件，并归档两个run的依赖清单。这六份源码的大小和MD5均匹配生产与续训的两份清单。三个候选入口中，所选的根key生成语句与混合器构建函数AST完全一致。因此，这部分随机流机制不依赖于在这三个入口间做猜测；**实际执行入口及导入路径仍未确认**。

<div id="historical-key-evidence-placeholder"></div>

### `None`和0不是同一个配置

历史入口使用下面的逻辑，见[EP入口](sources/historical_key_2026_10_08/train_ep.py)的808–810行，以及251行开始的混合器构建：

```python
data_key, model_key = jax.random.split(jax.random.PRNGKey(trainer.seed), 2)
if config.trainer.data_seed is not None:
    data_key = jax.random.PRNGKey(config.trainer.data_seed)
mix_key, shuffle_key = jax.random.split(data_key)
```

最后一行在数据集构建函数内；上面合写是为了展示依赖关系，不是原文的连续四行。`shuffle_key`再经[原key迭代器](sources/historical_key_2026_10_08/jax_utils.py)不断拆分，按训练数据集的插入顺序分给各桶。

公开配置声明训练seed为0、`data_seed=None`、`jax_threefry_partitionable=True`、数据模式为`MIXTURE`。在本轮明确使用`threefry2x32`的CPU环境中：

- 默认`None`：先取根key拆分的第一个作为数据key，得到`[1797259609, 2579123966]`；混合器key为`[4165894930, 804218099]`。
- 显式0：覆盖成`PRNGKey(0)`，即`[0, 0]`；混合器key变成`[1797259609, 2579123966]`。
- 两者模型key相同，但200个训练桶的child key全部不同，合成混合块前缀也不同。

这不是历史代码中的bug。相反，原来的`is not None`保留了0的合法含义。本轮额外把这一处条件改为`if data_seed:`做反事实控制：在训练seed为1、显式数据seed为0时，错误写法会跳过覆盖，重新跟随训练seed。这个错误版本没有出现在已核查源码中，不能写进Marin事故清单。

### 依赖版本核对后的结论更具体了

两个run的依赖清单都声明`jax==jaxlib==0.11.0`、`numpy==2.3.5`；代码artifact中的JAX版本文件也写明release为0.11.0。本轮新建独立CPU环境匹配这几个版本，保留前轮JAX 0.7.2/NumPy 2.5.3环境做比较。

每套环境执行9项控制、8组seed/数据seed/partitionable组合，导出每组200个child key。两套环境的8组结果、1600行key记录、合成混合块前缀摘要与逐桶计数全部一致。这里可以说“所选随机流控制没有复现版本差异”，不能说“两个JAX版本训练等价”。未比较完整模型、GPU编译、全部随机算子或实际运行的环境变量。

将`jax_threefry_partitionable`从True改成False时，根拆分结果和200个child key会改变，合成块前缀也改变。这个flag同时参与所用随机路径，本轮没有把key变化和块置换实现的贡献拆开。故复现所需的信息应包括PRNG实现与相关配置，不能只记录整数seed。JAX对key表示、拆分与PRNG实现的说明见[官方随机数文档](https://docs.jax.dev/en/latest/jax.random.html)。

### 用配置派生key后，部分块问题还在不在

V121/V122用直接给定的混合器key做边界控制。V123改用上传入口生成的数据key，再执行原混合器构建与child key分配。对子数据集仍使用有限身份替代，在生产配置与c00删域续训配置之间复算block8、切点9216：

- 两个前缀各有7938个逻辑身份未匹配。
- 生产前缀与续训配置下同块剩余部分相交5346个逻辑身份；下一批1024槽相交129个。
- 逐桶前缀计数差的半L1为2574；总槽数仍守恒。

这些数在两个选定CPU版本上完全相同。它们是**声明配置 + 上传源码 + 本轮指定PRNG环境**下的合成身份结果，替代了随意给定一个混合器key的控制条件；不是从真实历史进程读出的key，更不是5346条重复文档或重复token。只复算了这个生产/续训配置对，没有扩大为80个续训都重建了实际随机流。

[两版本比较与原值](analysis/historical_key_versions.json)、[0.11.0完整控制](analysis/historical_key_pipeline_jax_0_11_0.json)、[0.7.2完整控制](analysis/historical_key_pipeline_jax_0_7_2.json)、[逐桶key导出](analysis/historical_child_keys_jax_0_11_0.csv)。子数据集的真实block shuffle、模拟库存切片和缓存内容没有执行；记录的是分配给它们的key。

还有一个恢复分支必须同时考虑：EP入口844行先以`model_key`初始化状态，848行开始再将`state`交给checkpoint恢复，返回值覆盖该变量；辅助恢复函数在读取成功后返回`loaded`。因此，**初始化key改变不证明续训模型参数改变**。完整恢复、部分恢复、找不到checkpoint而从头初始化，是不同情形。本轮只检查源码顺序，没有执行这些模型恢复分支；前述key控制不能当作实际初始化方差实验。

### 怎样设计自己的实验

1. **先说清楚要估计哪种波动。** 默认数据seed跟随训练seed时，换seed会同时改变模型初始化key和数据key。从头训练时，这可能同时引入模型初始化与数据顺序变化，不能直接解释为纯模型初始化方差。只想看模型随机性，应在从头设计实验时，让所有候选及基线共享显式数据seed；想看数据顺序波动，则固定模型key并单独改变数据流。后者需要支持独立模型key的入口或受控改动，当前审计不声称已有该配置开关。
2. **接续已有训练，不能为了“规范”把空数据seed补成0。** 这会切换数据流。先记录原派生key、PRNG实现、相关flag、桶顺序与缓存身份；若决定换流，明确把它作为干预。新实验统一数据seed也要包含重新跑的基线，不能拿改变后的候选直接对旧默认流基线归因。
3. **保存随机流的结构。** 配置schema应区分缺省、`None`和合法0；评审记录拆分路径、命名数据桶到key的映射、混合块坐标与样本身份摘要。只存某个seed字段，无法检出多拆一次key、改桶顺序或修改flag。
4. **声明派生不等于执行证明。** 实际入口、导入文件摘要、执行的key和输入身份一旦可取得，应与本轮派生结果逐项对照；匹配之前，不把CPU身份控制升级成历史训练事故。

新增[随机流评审模板](templates/random_stream_review.json)，为空且未执行。可用`make historical-key-replay CPU_PYTHON=/path/to/python-with-jax`生成当前运行时版本的控制结果；分别运行两套环境后，再执行`make historical-key-compare`。两个环境都要独立保留，不能升级前轮环境后声称做过跨版本比较。
