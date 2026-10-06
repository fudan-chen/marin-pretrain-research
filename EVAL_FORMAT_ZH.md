# Paloma 评分格式与连续 token 流

PTB的loss下降占据了最近一次等权域均值改善的大部分，值得追查；但不能直接推出“应该加新闻数据”。先确定评分对象：文档如何切成序列，哪些位置实际计分，每个checkpoint是否收到同一批输入。这里有一个容易被配置名称掩盖的差别：固定源码中的文本`pack=False`仍然读取连续token流，并非每篇文档独立评分。

## 论文规定与实际实现

Paloma最终论文的G5及附录C.2.3规定：文档单独评估，长文档无重叠切分。附录H报告了拼接方式与子采样波动的初步观察，作者同时指出随机种子改变等可能因素。这不是对所有模型的因果证明，也没有测量Marin。论文按域设计评估，不能把其中的采样方差直接作为Hero曲线的误差条。[NeurIPS 2024原文，第5、24至25、37页](https://papers.neurips.cc/paper_files/paper/2024/file/760b2d94398aa61468aa3bc11506d9ea-Paper-Datasets_and_Benchmarks_Track.pdf)

数据卡列出了`ptb`与`twitterAAE_HELM_fixed`，各有`val`、`test`；Marin配置中的split则为`validation`。公开卡片能定位来源名称，不能证明哪一份split被加工进Marin缓存。此次归档固定HF revision，保存的是公开卡片与元信息，没有取得受限数据。[Paloma数据卡](https://huggingface.co/datasets/allenai/paloma/blob/65cd6fc59dba021b21db414fa5e8d7765ffbe5e6/README.md)

|核查对象|已看到的证据|能据此作出的判断|
|---|---|---|
|最新声明配置|两域均为文本格式，`pack=None`，attention跨文档屏蔽为true|对应固定源码的文本默认分支；历史执行SHA仍未取得|
|`_effective_pack`|文本格式默认返回False|不要把False解释成“每篇文档独立一个样本”|
|`_build_dataset_from_cache`|False分支构造`CausalLmDataset(TokenSeqDataset(...))`|先从缓存的平坦token字段按固定长度取窗口，再构造LM例子|
|`TokenSeqDataset.async_len`|平坦`input_ids`长度除以序列长度，向下取整|余数不进入该逻辑数据集；这发生在loader处理最后一批之前|
|真实缓存ledger|两个匿名HTTPS请求均为403、AccessDenied|本次没有读到ledger；不能据403断言对象存在，也不能确定区域或拒绝原因|

源码见[固定数据构造实现](sources/deepening_2026_10_04/datasets_production.py)，访问边界见[请求记录](analysis/cache_access_2026_10_06.json)。ledger路径依据固定cache代码及配置推导为`validation/shard_ledger.json`；它仍需由实际存储布局确认。

## 人工缓存如何暴露窗口差别

原类的方法在CPU上接受一个人工平坦缓存。两篇文档分别为`[10,11,EOS]`和`[20,21,22,23,24,EOS]`，EOS用0表示，窗口长度取4。这里检查的是输入切片，没有执行模型、attention或loss。

|排列|原方法输出的窗口|未进入逻辑样本的余数|
|---|---|---|
|A后B|`[10,11,0,20]`；`[21,22,23,24]`|B的最后一个EOS|
|B后A|`[20,21,22,23]`；`[24,0,10,11]`|A的最后一个EOS|
|仅有3个token的缓存|零个完整窗口|三个输入位置全部留在余数中|

同一组文档换顺序，窗口边界就变了。即使attention阻止B读取A，B的首token仍可能落在上一个窗口，下一窗口中的21无法读取20。屏蔽文档间attention处理的是可见关系，固定窗口处理的是输入覆盖和上下文截断，两者不能互相替代。EOS是否计分、窗口首位如何移位、padding权重是多少，还须沿LM例子和loss实现核查，不能把“余下一个输入位置”写成“少计一个有效目标”。相关区分见[文档边界](DOCUMENT_BOUNDARIES_ZH.md)。

13项检查验证了原方法的窗口、长度、余数、越界处理及声明配置。人工cache只模拟平坦字段接口，未读取TreeCache真实存储，也未运行GPU。这证明了需要检查的机制，不能证明Hero的PTB波动来自切分，更不能证明Hero实际遗漏了多少token。[脚本](scripts/probe_eval_format.py)及[结果](analysis/eval_format_probe.json)

## 对配比实验的影响

一个固定窗口格式与论文不同，并不自动成为bug。训练监控可以有自己的稳定口径；问题在于切换口径后继续沿用旧曲线名称，或把一个口径的改善解释成另一个任务的收益。

|准备作出的结论|先取得什么|达标后如何继续|
|---|---|---|
|“PTB变化来自模型更新”|两checkpoint的同一组输入数组、mask、byte表，逐批N/T/B，参数视图与执行身份|按现有导出器和比较器分开检查输入身份与数值差异|
|“PTB变化来自新配比”|同起点状态、固定评估格式、相同训练预算，配比对照分支|再比较各域改善与受损域；PTB均值贡献不能替代反事实|
|“加新闻能改善整体能力”|新闻增量的实际曝光、去重后覆盖、冻结的未参与选择的评估集|确认预算从哪里扣，检查供体受损和跨域迁移，而不只看PTB|
|“扩大评估集能解决波动”|同checkpoint上分组重评，分组单位及上下文构造固定|估计本域差值的不确定性；不要搬用论文在另一模型/格式上的方差|

建议执行顺序是：先冻结现有评分管线并收集真实逐批记录；再用同一checkpoint做文档独立格式的旁路评分；两条指标分别命名、分别建基线。比较格式时还要记录不同的有效目标、上下文长度和窗口边界，不能声称只改变了一个布尔开关。最后才开启配比或顺序实验。这样才能区分模型变好、评分对象改变与预算重新分配。

默认评估重新遍历有限数据集的源码证据已经在[评估身份核查](EVAL_IDENTITY_ZH.md)中确认。因此，在当前证据下，“每次随机抽不同样本导致PTB变化”不能作为默认解释。固定输入上的训练变化仍可能使loss波动；缓存是否被改写、历史代码与配置是否一致，则需要真实摘要才能排除。

可直接执行的管线为：真实全局数组 → [离线导出](EVAL_ARRAY_EXPORT_ZH.md) → [逐批对照](EVAL_REPLAY_ZH.md) → 固定格式旁路重评 → 配比反事实。前两步已有离线工具；真实数组、旁路评分和配比训练仍待取得。此处没有新增训练收益或PTB根因结论。

公开来源的本地副本见[固定卡片](sources/paloma_protocol_2026_10_06/dataset_card.html)、[最终论文PDF](sources/paloma_protocol_2026_10_06/paloma_neurips_2024.pdf)与[取得记录](analysis/paloma_protocol_acquisition.json)。
