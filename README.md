# HiddenDetect复现

## 当前入口（2026-11-05）

查看同一棵实验导图和对照表：运行 `python viewer/build.py`、`python -m http.server 8000`，打开 `http://localhost:8000/viewer/`。[查看页说明](viewer/README.md)。仅需Python标准库；不需要下载模型或论文库。

- [实验编号与历史追溯](docs/experiment-numbering.md)：质心已分配EI-2-7／EI-3-1，位置与选层独立为CFG；历史32层结果与待验证14层方案分开。
- [完整问题树](research-map.md)与[唯一方向登记](docs/direction-registry.md)。本分支整理当前维护文档，原冻结运行和指标不改。
- [EI-PA2解除PCA32限制的结果](results/ei2-pca-uncapped/report-2026-11-05.md)：549次LR、24门训练及独立核验完成；未确认稳定总体AUROC增益。
- [EI-2-3-4三组组合验证](results/ei2-pca-joint/report-2026-11-05.md)已完成EI-PJ1历史匹配验证及独立复核；组合2最值得独立确认，组合3在OR退步。质心C14两方案仍仅登记，未产生新成绩。

下方按时间保留历史结果；旧EI-1质心、EI-3位置、EI-4选层必须结合上述新编号追溯，不与新版EI含义混用。

2026-11-04 OC-O1原OC-4数据验证完成：原train1600/dev600/cal400，RV1082原积分AUROC0.756547；两组八进阶变式重新拟合，144拟合、38模型重载、84万独立ROC与54差区间复算。C1000补充，保留H1旧头与本轮匹配基线。历史诊断，无独立新确认。[报告](results/oc4-original/report-2026-11-04.md)、[解释](docs/oc4-original-findings-2026-11-04.md)。

2026-11-04 OC-A2注意力改进完成：原14F，训练300、5折、全批240更新；11神经头共199拟合，五项四级候选与机制消融、三种子，15预设比较；33模型重载、39指标组及13万独立ROC全部复核通过。历史诊断，无独立新确认。[报告](results/oc-attention/report-2026-11-04.md)、[解释](docs/oc-attention-findings-2026-11-04.md)。

最新：2026-11-04 OC-H3三组检测头重训完成，旧完整1600/匹配旧300/OR300，18新头与三种子。同批SVM检出69.4%→76.4%，误报39→40；相对完整旧头的pAUC改善未确认，相对匹配旧300获区间支持，仅历史诊断。[报告](results/oc-retraining/report-2026-11-04.md)。

**2026-11-04 OC-H2跨来源确认与稳健性完成：** OR-Bench测试1000正常/500危险，另400正常校准；固定B、逻辑回归、线性SVM、注意力汇总，完成15次训练重采样和1000次校准重采样。SVM相对B及LR的低误报pAUC获本来源支持；总体AUROC点值仍以B最高，注意力机制门槛未通过。[完整报告](results/oc-confirmation/report-2026-11-04.md)。

**2026-11-04 OC-4八检测头验证完成：** 固定同样14层F，8方案×3候选×3种子，共72拟合。线性三头与MLP在两批历史上的主pAUC差区间为正；复杂头未稳定超过逻辑回归，无独立新确认。24模型复算及独立10000次统计审查通过；F、逐条分数和权重留校，本地仅交付汇总。[完整报告](results/oc-heads/report-2026-11-04.md)。

**2026-10-30 OC-2/3历史验证补齐：** 三种单层评分与固定层差完成；均值主改善未获支持，其余新增规则更差。仅余10正常未做独立新确认；424统计双端和独立审查通过。[完整报告](results/oc23/report-2026-10-30.md)。

**2026-10-30 OC-1拒绝词参照完成：** 清理、扩充、加权三子实验新1082确认及独立复核完成；清理低误报改善未确认，扩充/加权主差区间全负，保留原RV。已有四汇总实验映射为OC-3；[完整报告](results/oc-reference/report-2026-10-30.md)。

**2026-10-30选层首轮实验与独立复核完成。** 固定原F、模板末位和梯形积分，在256条选择集上选层，再评估去除22872条历史输入及近重复后的2000条新同源确认。默认[16,29] AUROC **0.776240**；同预算FDV[12,28] **0.801085**；496区间直接AUROC最大[9,19] **0.829589**；重采样下尾AUROC[8,20] **0.828034**。两种新选层方法对默认均有修正区间支持；重采样相对直接搜索的额外收益未确认。[完整结果](results/layer-selection/report-2026-10-30.md)、[验证记录](results/layer-selection/verification.json)、[文献与原方案](docs/layer-selection-survey-2026-10-30.md)。分支`research/layer-selection-survey`，正式方向ID待指定。

**2026-10-30纯OC补实验完成：** 固定原末位F与14层，仅改汇总。新同源2000上等权M和截尾相对原积分有小幅低误报排序增益，top4/连续窗口未确认；截尾未确认超过M且误报增加。保留B部署参照，优先后续验证M。273统计双端与独立审查通过；[完整报告](results/pure-oc/report-2026-10-30.md)。原EI结论保留。

**2026-10-30原规则EI-3位置补实验及新确认完成，保留模板末位。** C上末位0.746640，六用户点均更低，用户末位0.670976；模板倒数2为0.773488、倒数3为0.632636。倒数2经新同源1000确认：同批末位0.771068→0.786908，差95%区间[−0.0045095,+0.0363221]含0，pAUC下降，召回同29.8%、误报4.8%→6.2%。无分类器、无重选层。三运行双端258/98/58个统计数值差0，独立审查通过。[用户报告](results/original-positions/report-2026-10-30.md)、[模板报告](results/original-template/report-2026-10-30.md)、[新确认与结论](results/original-template-confirm/report-2026-10-30.md)、[唯一登记](docs/direction-registry.md)。分支`experiment/original-rule-positions`；下方为历史交接。

**R28—R29少量／边界融合验证完成：** 固定R27 draw0四个岭头，用独立dev600作元训练，普通／动态／同维随机每策略各9候选，并增加同2200标签预算的纯末位重训练对照。新Aegis1000：B AUROC0.923892／pAUC0.797053，全局动态0.925028／0.799495，局部动态0.924184／0.798589；局部只改15.9%分数，但召回68.2%→66.6%，误报5.0%→4.8%，七比较校正后未确认增益。新Toxic150毒性辅助也未确认局部收益。预选仍B。640项CV、66条件和bootstrap双端重放通过。[完整报告与图](results/anchored/report-2026-10-29.md)。分支`experiment/anchored-summary`。

**R27训练稳定性验证完成：** 原1600训练＋新1600构成池，五组各抽1600；岭回归／逻辑回归×末位／均值／动态／同维双随机，共40模型、3600折。新1000确认中普通均值的低误报pAUC仅1/5组超过末位；动态对随机两头均5/5正向，但六项校正区间仍含零。新确认岭回归B／均值／动态AUROC均值0.914886／0.915766／0.916454，召回69.72%／67.04%／66.60%、误报5.88%／5.04%／5.04%；没有确认稳定补强末位。双端3600折、160条件及bootstrap重放通过。[完整报告与图](results/stability/report-2026-10-29.md)。分支`experiment/training-stability`，五训练均值不是集成模型成绩。

**R23—R26检测头替代验证完成：** 固定全部32层模板末位131072维，比较L2逻辑回归、标准线性SVM、RBF核岭回归，岭回归也给予相同9组参数。360折指标、35项测试条件及bootstrap双端重放通过。新1000确认AUROC：岭回归0.913120、逻辑回归0.912956、SVM0.900180、RBF0.913496；替代均未确认增益，SVM排序变差。逻辑回归／RBF召回提升伴随误报点值增加，最终仍保留岭回归。[完整报告](results/heads/report-2026-10-29.md)。分支`experiment/detector-heads`，固定基准v1不变。

**R19—R22用户信息加工已完成：** 将全部有效用户token加工为普通均值、动态注意力／变化摘要、跨层与模块增量、局部—整体关系。13个冻结候选，91项指标、130折及5000-bootstrap两端重放通过。新1000确认AUROC：B0.919636、均值0.922112、动态0.922460、增量0.921720、关系0.923084；出现小幅正向线索，但十项校正后尚未确认增益，也未证明复杂加工胜过普通均值。CV按预定规则仍选B。[完整报告与图](results/processing/report-2026-10-29.md)。分支`experiment/user-signal-processing`，固定基准v1不变。

**R16—R18已完成：** 全量固定512控制、两个模板邻位、32层逐层用户／模板末位选择。五种算法各5种子×64唯一掩码，59头、413项指标、11460个CV折指标及bootstrap双端复核通过。新1000确认：末位0.927328，替换32层0.927056，替换27＋32层0.927008；无可靠AUROC增益，保留原末位。简单双层扫描比群体算法更稳定找到CV最佳组合。[完整交接](results/routing/report-2026-10-29.md)。分支`experiment/layer-readout-routing`，固定基准v1不变。

**R14—R15真实位置实验已完成：** 保持32层完整向量和同一ridge，比较9个真实token读出及8个双点对照，170次训练折拟合、136项指标复核通过。新1000条确认中模板末位AUROC0.92729，用户末位0.90900，二者组合0.92786但增量区间含零；预设近似不劣上界标准未通过，不能宣称全局最优。[完整交接](results/positions/report-2026-10-29.md)。分支`experiment/token-positions`，固定基准v1不变，末位权重与旧B完全一致。

**R11—R13已完成：** `experiment/swarm-selection`固定[统一基准v1](configs/benchmarks/v1.json)，完成GA／DE／PSO／随机五种子、每次120唯一掩码的公平比较，并另测400条未使用输入。GA开发集增益未泛化：独立确认AUROC均值0.90052，末位0.90380；DE、PSO、随机及L1均退回末位。48方案、336项复核通过，旧模型22080个分数完全一致。[完整报告](results/swarm/report-2026-10-29.md)。

实验轮次现按每次实质方法改动统一为R0—R29，避免把运行批次当作优化轮次。[完整轮次、数字归属与可比性](docs/experiment-rounds.md)。其中0.87402属于全部F/C融合，0.92837属于未投影末位完整向量，两者均不是PSO的成绩。

**2026-10-29：条件融合改进轮已完成。** 新清单2824条，23候选先冻结后独立评估，138项组合复核通过。主测试加性融合AUROC0.91920，强末位0.91576，存在小幅排序增量；既定阈值召回未提高，显式交互暂无额外收益。ToxicChat来源安全校准将强末位召回30.0%提高至55.5%，同时误报1.0%增至4.5%，AUROC不变。[完整报告](results/fusion/report-2026-10-29.md)。本轮分支`experiment/conditional-fusion`；SSH中断已恢复，旧结果与分支保留。

**第二轮高维实验已完成：** 3700条新输入、259组／19,270个投影与统计特征，47个方案先冻结后独立评估。未投影末位ridge AUROC0.92837，多位置合并0.92606，增量区间包含零；分组PSO只有小幅同任务收益，未超过强末位基线。完整图表、协议、恢复记录与限制见[第二轮报告](results/highdim/report-2026-10-28.md)。工作分支`experiment/highdim-interactions`；原两轮基线和历史分支保留，数值不跨不同测试划分直接比较。

**互补信号与BPSO实验已完成：** 新4800条数据按训练／搜索／校准／测试隔离，19个候选先冻结后评估。Aegis监督F32 AUROC0.83740，融合隐藏状态质心后0.87402，完整下一token logits探针0.87822；BPSO五种子平均0.87075，等预算随机0.87078，未见稳定算法收益，外部召回也有下降。所有种子、置信区间和限制见[前瞻实验报告](results/signal-selection/report-2026-10-28.md)。本实验在`experiment/signal-selection`分支，保留首轮基线。

**2026-10-28首轮实际结果：** 8条文本冒烟通过，450条XSTest默认区间AUROC为0.85828（论文0.868，尚未对齐）。作者12条原校准选出0基[17,29]，对应同一文本测试AUROC为0.85082。原信号、日志、环境和校准重采样已归档，见[统一结果报告](results/baseline-summary-2026-10-28.md)。尚无多模态最终测试指标。

目标：先确认作者原方法在明确设置下能运行、结果是否接近正式论文，再检查选层稳定性和误报／漏检。首轮不引入RLDN-AOO，不同时设计新攻击、新防御和benchmark。

## 入口

- [真实进度与下一步](progress.md)
- [复现协议与阶段边界](protocol.md)
- [正式论文PDF](https://aclanthology.org/2025.acl-long.724.pdf)
- [正式ACL条目](https://aclanthology.org/2025.acl-long.724/)
- [作者代码](https://github.com/leigest519/HiddenDetect)

本仓库是独立复现记录，不是作者官方实现。论文阅读笔记与相关工作调研保留在本地Paper项目，不随本仓库发布。本次新增不发布训练权重；历史已公开的小检测头和自制图按原记录保留。

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

仓库保留真实运行日志和失败记录。`source/`、`data/`、`models/`、`env/`及本地缓存均不纳入Git，下载清单、revision和SHA256记录在configs中。作者代码和权重的许可分别适用，不因本复现仓库公开而取得额外分发许可。本仓库不包含作者源码或大模型权重；历史已公开的小检测头按原记录保留，本次新增训练权重不上传。

在学校实验目录中运行以下命令会执行真实检测，须确保模型与数据已按锁定清单准备：

```bash
env/bin/python configs/run_text.py
env/bin/python configs/run_text.py --manifest configs/xstest450-manifest.json --output results/xstest450-default-s16e29
env/bin/python configs/run_selection.py
env/bin/python configs/diagnose_calibration.py
```

安装配置见`configs/bootstrap-inference.sh`和`configs/requirements-inference.txt`。PEFT0.7.1是已验证的兼容调整，作者原0.14冲突日志保留。环境省略7个UI依赖，实际模型检测通过，不承诺WebUI或训练栈可用。第三条命令会使用作者的12条原校准数据重新选层，不能用测试分数自行调整区间。最后一条只重采样缓存，不重新运行底座。

每个阶段更新`progress.md`并重建项目阅读网页。不得把静态代码核查、冒烟测试、论文指标复现或方法改进混写为同一种“完成”。

## OC-S1：SVM改进（2026-11-04）

四级编号三项已完成历史诊断，未确认优于同批调参线性SVM，RBF低误报pAUC下降。协议、242拟合、17模型重载与8万独立ROC复核见[报告](results/oc-svm/report-2026-11-04.md)。注意力仍待后续阶段。
