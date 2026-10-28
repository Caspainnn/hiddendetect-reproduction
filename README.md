# HiddenDetect复现

目标：先确认作者原方法在明确设置下能运行、结果是否接近正式论文，再检查选层稳定性和误报／漏检。首轮不引入RLDN-AOO，不同时设计新攻击、新防御和benchmark。

## 入口

- [真实进度与下一步](progress.md)
- [复现协议与阶段边界](protocol.md)
- [正式PDF，L042](../../literature/papers/HiddenDetect.pdf)
- [正式ACL条目](https://aclanthology.org/2025.acl-long.724/)
- [作者代码](https://github.com/leigest519/HiddenDetect)
- [P02阅读网页](../../readings/论文阅读/论文阅读记录.html#p02)
- [10篇相关工作及开源审计](../../readings/研究笔记/内部信号安全检测与HiddenDetect特征选择-10篇调研-2026-10-27.md)

## 目录

|目录|用途|
|---|---|
|source/|完整作者仓库已固定到2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb，含作者自带小规模数据|
|configs/|环境清点脚本及尚未执行的冒烟配置，抽样ID待数据版本确认|
|data/|后续外部数据或路径说明，当前为空；作者自带数据在source中|
|models/|目标机器模型路径说明，当前为空，不复制大模型|
|results/|第一阶段静态审计报告，无检测性能结果|
|logs/|本机环境、torch实际导入、文件SHA256与校准压缩包清单|

计算条件：用户确认学校RTX 4090 24GB、SSH、Linux、60GB可用存储，有Conda但无模型权重，SSH待配置，尚未远程实测。本地RTX 3060 Laptop 6GB只作辅助，当前Python的torch为CPU版。代码可公开读取，锁定树内未发现独立许可声明，保留本地研究副本，不对外再分发。

每个阶段更新`progress.md`并重建项目阅读网页。不得把静态代码核查、冒烟测试、论文指标复现或方法改进混写为同一种“完成”。
