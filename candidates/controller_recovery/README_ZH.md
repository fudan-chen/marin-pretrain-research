# 控制器恢复候选：研究代码，尚未接入

以固定 `eee467718515b2383fc3a433014afce4ab075b05` 的完整原 checkpoint.py/main.py 为底稿。原作者版权头保留；改动见 proposal.patch，生成脚本为 scripts/build_controller_recovery_candidate.py。没有改上游 checkout 或运行生产服务。

候选删除请求回滚入口的预先清空。发布必须同时备份主库与 auth；上传后最后写 checkpoint.complete.json，列出格式、epoch、两份压缩文件的尺寸与 SHA256。完成记录写入成功后，才更新本地 ancestry。选择最新时跳过没有完成记录的目录；如果有目录却没有任何完成记录，报错，不当成首次启动。下载先检查记录身份、完整文件、尺寸/摘要，再执行原解压与 SQLite 检查，最后使用原 staging 替换流程。

22 项本地控制通过，使用真实 SQLite/zstd 和原完整模块的候选副本：坏/缺失恢复、回滚成功后只消费一次、第二份上传失败、完成记录上传失败、缺 auth、摘要改变、摘要正确但数据库坏、epoch 不匹配、相同 epoch 覆盖与旧格式拒绝。详细记录在 analysis/controller_recovery_candidate_controls.json。故障是构造的，存储为 file://。

这不是可部署的整套 Iris。只验证候选 write_checkpoint/prepare_controller_state 等路径；原 Controller.begin_checkpoint 的生产调用链没有改成这个候选。若只换启动入口，仍可能由旧写入路径产生无完成记录备份，随后被候选拒绝。因此接入必须覆盖所有写入者和读取者，再检查并发与迁移；不能只替换两个文件就发布。

候选主动拒绝没有完成记录的旧格式，包含原上游可读的旧格式；这是一项未完成的兼容工作。需要明确迁移来源、所需 auth 状态、校验和失败保留策略，不能自动用新 auth 文件充当恢复。文件齐全不等于两个库在同一事务快照，SHA 不等于签名认证。本轮没有验证强制杀进程、云存储、并发发布、旧签名密钥、schema 迁移策略、模型训练 checkpoint 或 loader 游标。

复查命令应在冻结上游 checkout 的已同步环境运行：

```sh
uv run --frozen --no-sync --package marin-iris --group test --no-default-groups python /Users/chen/Documents/ChatGPT/marin-pretrain-research/scripts/probe_controller_recovery_candidate.py
```

研究结论可以采用，生产候选继续保留。应先完成调用链接入和旧格式迁移，再用真实中断、同 epoch 并发、对象存储故障与状态恢复对照决定是否值得合并。

V105 范围复核：当前原 auth 数据库没有业务表，签名密钥来自配置 secret；两文件要求不能直接解释为保护当前签名密钥。人工跨文件 generation 控制显示候选会发布并接受 main=0/auth=1 的有效数据库对，因此完成记录不保证共同 snapshot。这个反例不是生产 auth 状态或真实训练状态；候选未因此升级保证。
