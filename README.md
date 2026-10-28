# HiddenDetect复现

**2026-10-28首轮实际结果：** 8条文本冒烟通过，450条XSTest默认区间AUROC为0.85828（论文0.868，尚未对齐）。作者12条原校准选出0基[17,29]，对应同一文本测试AUROC为0.85082。原信号、日志、环境和校准重采样已归档，见[统一结果报告](results/baseline-summary-2026-10-28.md)。尚无多模态最终测试指标。

目标：先确认作者原方法在明确设置下能运行、结果是否接近正式论文，再检查选层稳定性和误报／漏检。首轮不引入RLDN-AOO，不同时设计新攻击、新防御和benchmark。

## 入口

- [真实进度与下一步](progress.md)
- [复现协议与阶段边界](protocol.md)
- [正式论文PDF](https://aclanthology.org/2025.acl-long.724.pdf)
- [正式ACL条目](https://aclanthology.org/2025.acl-long.724/)
- [作者代码](https://github.com/leigest519/HiddenDetect)

本仓库是独立复现记录，不是作者官方实现。论文阅读笔记与相关工作调研保留在本地Paper项目，不随本仓库发布。

## 目录

|目录|用途|
|---|---|
|source/|完整作者仓库已固定到2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb，含作者自带小规模数据|
|configs/|固定模型与数据清单、环境安装、文本运行与原选层脚本及轻量验证|
|data/|原作者XSTest固定CSV，远程另有解压后的原12条校准数据，均不入Git|
|models/|约16GB模型文件只在学校服务器，锁定清单在configs中，本机不复制大模型|
|results/|8条冒烟、450条默认分数、12条原选层、32层信号与校准重采样|
|logs/|真实环境、下载、兼容错误、推理与验证日志，远程账户路径已脱敏|

计算条件：已通过SSH实际使用学校Ubuntu20.04.6和RTX4090D 24564MiB，容器内存上限60GiB，根盘初查47GB空闲，家目录共享存储个人配额仍待独立确认。复现有独立Python3.10环境，现有环境未改。本地RTX3060Laptop6GB只作辅助，当前本机torch为CPU版。代码可公开读取，锁定树内未发现独立许可声明，保留本地研究副本，不对外再分发。

## 查看代码与重跑

学校服务器目录为`~/HiddenDetect-reproduction`，可用VS Code Remote-SSH打开。GitHub仓库为[Caspainnn/hiddendetect-reproduction](https://github.com/Caspainnn/hiddendetect-reproduction)。`main`保存首轮基线快照，`reproduce/smoke`保留复现工作分支。作者源码在`source/HiddenDetect`，LLaVA在`source/LLaVA`，两者独立锁版本并从本仓库忽略。此目录只同步实验材料，论文PDF和阅读网站仍在本机Paper项目。

仓库保留真实运行日志和失败记录。`source/`、`data/`、`models/`、`env/`及本地缓存均不纳入Git，下载清单、revision和SHA256记录在configs中。作者代码和权重的许可分别适用，不因本复现仓库公开而取得额外分发许可。本仓库不包含作者源码或模型权重。

在学校实验目录中运行以下命令会执行真实检测，须确保模型与数据已按锁定清单准备：

```bash
env/bin/python configs/run_text.py
env/bin/python configs/run_text.py --manifest configs/xstest450-manifest.json --output results/xstest450-default-s16e29
env/bin/python configs/run_selection.py
env/bin/python configs/diagnose_calibration.py
```

安装配置见`configs/bootstrap-inference.sh`和`configs/requirements-inference.txt`。PEFT0.7.1是已验证的兼容调整，作者原0.14冲突日志保留。环境省略7个UI依赖，实际模型检测通过，不承诺WebUI或训练栈可用。第三条命令会使用作者的12条原校准数据重新选层，不能用测试分数自行调整区间。最后一条只重采样缓存，不重新运行底座。

每个阶段更新`progress.md`并重建项目阅读网页。不得把静态代码核查、冒烟测试、论文指标复现或方法改进混写为同一种“完成”。
