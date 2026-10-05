# 缓存、分词器与训练样本：配比比较还缺哪一层身份

本轮发现会改变跨规模实验的解释：六个公开ladder配置不仅改变了权重，还引用了两个不同的缓存根目录。旧组200桶全部来自`store_81e7e39a`，mix25组三个规模的200桶全部来自`store_4d2e363d`。桶名相同不能证明底层样本相同；目录不同也不能证明内容一定改变。因此，目前可以比较完整配方的观测端点，不能把差异全部归给配比切换。

这是对[跨规模研究](SCALE_TRANSFER_ZH.md)可比性检查的补充。它不改变已有loss数值，也不把未读取的缓存说成坏数据。

## 1. 六份配置实际用了什么

[逐桶审计JSON](analysis/cache_audit.json)保留1200条component记录及六份原配置摘要，审计脚本为[scripts/audit_cache.py](scripts/audit_cache.py)。没有从三个样例外推到全部桶。

| 实际字段 | 覆盖范围 | 应怎样理解 |
|---|---|---|
| 200个桶名，六个run | 1200条component | 同一个桶名是分类身份，不是内容摘要 |
| 旧组缓存根`store_81e7e39a` | 三个规模各200桶 | 与mix25组的缓存地址不同 |
| mix25组缓存根`store_4d2e363d` | 三个规模各200桶 | 是否只是重写目录、是否改变过滤或分词，尚无实际ledger/token证据 |
| `format={text_key:text}`、`pack=null` | 1200条 | 按归档源码的文本分支进入连续流固定长度切片 |
| `flat_cache=true`、`source=null` | 1200条 | 直接读取现有缓存；flat_cache描述split目录布局，不能单独用它推断attention规则 |
| tokenizer名称`marin-community/marin-tokenizer` | 六份配置 | 名称相同；配置名称本身没有固定Hub commit |
| `block_cross_document_attention=true` | 六份配置 | 开启边界attention规则；实际边界依赖读出的token，不能靠这个开关证明EOS已经写入 |
| component的`split=validation` | 1200条 | 这个字符串不能单独证明训练使用了验证集。归档读取路径中，flat_cache对请求的train返回根目录，对其他请求返回None；须查真正的分割与调用链 |

相关路径见[dataset_for_component与缓存读取](sources/boundaries_2026_10_05/datasets.py)。以上源码固定在`84869ae8c91ffe64e9f761c5bd714542eb1876e0`，尚未绑定为这些历史run实际执行的版本。

## 2. enforce_eos不等于把旧缓存补上EOS

构建新文本缓存时，`BatchTokenizer`在关闭自动特殊标记的编码调用之外，显式在文本后附加EOS字符串，并在编码结果前插入BOS ID。预处理元数据记录名称、词表大小和append_bos/append_eos等字段。

读取现有缓存时，`load_lm_dataset_cache`构造预处理器，用它提供输出示例和预期元数据，再调用`TreeCache.load`。这条路径不会重新逐条处理现有input_ids。因此，读取时的`enforce_eos=True`不能当作缓存已经补齐EOS的证据。

```text
新文本 → BatchTokenizer显式处理边界 → 写入input_ids
旧input_ids → 比较预处理元数据 → 加载缓存 → 固定长度切片
                                      ↓
                     EOS实际存在时，才可用它推断文档边界
```

源码：[文本缓存入口](sources/cache_2026_10_05/cache.py)、[分词预处理](sources/cache_2026_10_05/batch_tokenizer.py)、[format分派](sources/cache_2026_10_05/formats.py)。`BatchTokenizer.metadata`未包含Hub commit或分词器文件的内容摘要。即使名称、词表大小都相同，也不足以排除同名分词器内容发生变化。

## 3. 元数据不同，源码会怎么做

`TreeCache.load_from_ledger`比较ledger元数据与预期元数据。出现差异时调用logger.warning，随后继续检查缓存是否完成。完成的缓存仍会进入构造器；未完成的缓存才在该分支抛出FileNotFoundError。`CacheLedger.load`也采用warning方式处理预处理元数据差异。

| 检查情形 | 原加载分支的行为 | 本轮验证范围 |
|---|---|---|
| 比较结果为空，缓存完成 | 进入构造器 | 执行原函数，替代文件系统与构造器 |
| 比较结果报告tokenizer差异，缓存完成 | warning后进入构造器 | 提供人工比较结果；没有运行真实DeepDiff |
| 比较结果报告元数据缺失，缓存完成 | warning后进入构造器 | 同上；不是实际生产ledger |
| 缓存未完成 | 抛出FileNotFoundError | 原条件分支 |

[TreeCache源码](sources/cache_2026_10_05/tree_cache.py)说明这是告警式校验，而非严格拒绝策略。本轮执行的是原加载分支，比较结果由替代依赖提供；没有验证真实存储内容，也没有证明线上出现过相同告警。

另一个需要接上的接口：[bucket_writer.py](sources/cache_2026_10_05/bucket_writer.py)写出的ledger使用`CacheMetadata.empty()`。Datakit store将分块token重新组装、过滤后写入桶，不能据此假定它重新执行了训练侧的BatchTokenizer。当前源码的空元数据写入与告警式读取可以组合成一条容许缺少分词身份的路径，但两个历史store的实际ledger和构建版本仍未知。不能把这条可能路径写成Hero故障的根因。

## 4. 特殊标记的当前身份与历史身份

本轮匿名读取Hub元数据，再以观察到的commit固定下载两份配置文件。当前固定版本为`a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`，不是已确认的历史训练版本。[抓取账本](analysis/cache_acquisition.json)保留URL、时间、长度与SHA256。

| 标记 | 当前配置中的ID | 解释边界 |
|---|---:|---|
| begin_of_text | 128000 | BOS身份；不证明缓存每条记录都包含它 |
| end_of_text | 128001 | EOS身份；不能用聊天轮结束标记替代 |
| finetune_right_pad_id | 128004 | 已登记的特殊标记；不证明本次预训练采用padding |
| eot_id | 128009 | 与EOS是不同ID，不能凭名字把二者混为文档结束 |

未下载执行完整tokenizer模型，也未读取S3缓存token数组。特殊标记配置不能回答真实文本是否被截断、BOS/EOS是否重复、聊天模板是否执行；这些问题需要编码结果或缓存样本。

## 5. 为什么这会影响配比与loss的解释

连续流分支把缓存的input_ids展平后按序列长度读取，长度为`floor(N/S)`。最后不足S的尾部不会成为该分支的完整样本。文档边界还依赖token本身，不能仅靠原始行数还原。

[文档边界深读](DOCUMENT_BOUNDARIES_ZH.md)已经说明，隔离attention不自动取消跨文档预测目标。现在可以把问题进一步定位：如果短文档比例上升，边界密度可能随之改变；如果两个store采用不同边界标记，统计相同名义token预算，也可能改变上下文与有效目标的组成。这里的“可能”不能用库存比例或总loss曲线直接消除。

对配比实验的实际建议是，先固定缓存身份与样本构造，再比较权重。若底层内容必须更换，就把它列为独立干预，保留共同缓存下的权重对照。否则一个更低的loss可能混合了样本更新、tokenizer变化、文档边界变化和权重调整，无法知道下一轮应该复制哪项操作。

## 6. 缓存进入配比实验前的验收管线

以下是新增工程验收支线，18条rubrics和七阶段主管线保持原版本。

| 规则 | 必须留下的证据 | 未满足时的决定 |
|---|---|---|
| K1 同桶同内容需要证明 | 两个store逐桶记录/分块摘要、token数、构建版本与过滤规则 | 只能称完整配方比较，不归因于单独权重 |
| K2 分词身份必须固定 | Hub commit及实际tokenizer文件摘要，不能只记录名称与词表大小 | tokenizer等价未知 |
| K3 加载告警必须进入实验记录 | ledger与预期预处理元数据差异；明确兼容依据 | 拒绝用于严格配比归因，或另建已验证缓存 |
| K4 检查边界实际结果 | 从缓存读取的BOS/EOS位置、文档offset与序列切片对应关系 | 不声称attention隔离已按预期覆盖全部文档 |
| K5 单列目标计数 | 有效query、计分target、边界target和尾部余数 | 名义token曝光不能替代有效目标曝光 |
| K6 更换缓存独立列为干预 | 共同缓存权重对照，或缓存×权重交叉实验 | 不把二者合并解释为某种配比更优 |

执行顺序：冻结两个store身份 → 对照ledger → 抽取边界与切片 → 统计有效目标 → 共同缓存下调整权重 → 相同评估口径确认。当前停在身份与源码核查阶段；真实ledger、token数组、历史tokenizer commit、构建执行SHA均保留未知。
