# HiddenDetect 第一阶段核查（2026-10-27）

本阶段完成本机清点、作者仓库固定版本获取、运行入口及论文方法静态核查。未安装实验环境、下载外部数据或模型、运行作者代码或测量检测性能。作者仓库自带的少量数据随检出获取，与外部数据下载区分。

## 真实计算条件

|项目|实际核查或用户确认|
|---|---|
|本机系统|Windows 11，版本10.0.26200|
|本机GPU|RTX 3060 Laptop，6144MiB，驱动546.30；nvidia-smi所报CUDA 12.3是驱动支持上限，不是PyTorch CUDA运行时|
|本机Python|D:/Python/python.exe，3.12.1|
|本机PyTorch|2.3.1+cpu，cuda_available=False，device_count=0|
|本机磁盘|清点时C盘约10.95GiB空闲，E盘约173.69GiB空闲，精确字节见清单|
|包状态|未安装transformers、accelerate、llava、pyarrow、fastparquet；numpy 1.26.4、pandas 2.2.2、sklearn 1.5.1|
|学校机器|用户确认SSH、Linux、RTX 4090 24GB、60GB可用存储，有Conda但无模型权重，SSH尚未配置。以上尚非远程实测|

证据：[环境与文件清单](../logs/stage1-inventory-2026-10-27.json)、[PyTorch实际导入结果](../logs/local-torch-2026-10-27.json)。系统CIM查询遭权限限制，使用Python平台信息与磁盘API取得替代证据。默认PowerShell profile中的Conda hook出现DLL错误，后续检查均使用无profile shell，未修改用户配置。

## 作者版本与入口

仓库：https://github.com/leigest519/HiddenDetect 。本地位置：[作者仓库](../source/HiddenDetect/)。固定提交为 `2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb`，提交日期2025-06-08，已detached checkout，清点时工作树无修改。LLaVA依赖另固定到 `c121f0432da27facab705978f83c4ada465e46fd`。10个跟踪文件的本地SHA256已保存。Git在Windows检出可能转换CRLF，因此本地文件哈希与Git原始blob字节长度不必相同。

首次clone因127.0.0.1:7897失效代理失败，沙箱内直连又报getaddrinfo线程错误。随后通过允许的网络执行、仅对该命令禁用代理，成功clone和检出。未改全局代理。额外通过Python下载LLaVA pyproject的尝试被远端断开，已改用在线只读源码核验，不视为成功下载。

|入口|已核行为|本轮处理|
|---|---|---|
|code/test.py|主入口先构造7套数据，随后逐套调用test，每次重新加载模型，结果只写汇总CSV|禁止直接全量启动。后续单独加载XSTest和一次模型|
|code/safety_aware_layers.py|真实原选层入口，计算正常和不安全均值差FDV，打印大于最后层差值的层号，返回FDV|后续按论文Eq.4取这些层号的min/max形成连续区间，单独保存，不冒充已执行|
|code/test_qwen.py|另一底座分支，主入口使用21–24|不纳入首轮LLaVA实验|

原选层不是空白功能。此前审计快照只覆盖test.py与load_datasets.py，本次完整检出补齐了选层脚本和配套数据。源码没有把选层输出自动接到test.py，也未核到CogVLM运行入口。

## 论文与代码逐项核对

依据：本地正式PDF L042，§3.1–3.3、§4、§5.1及[正式ACL条目](https://aclanthology.org/2025.acl-long.724/)。文本核查辅助来自`.tmp/three-paper-review/HiddenDetect.pdf.txt`。

|核对点|确认内容与复现含义|
|---|---|
|层号|论文Eq.1使用0至L−1。源码遍历hidden_states[1:]去掉embedding项，F[0]对应第1个解码block之后，F[16:30]包含第17至第30个block输出，共14点，积分跨13个单位间隔|
|最后层特殊性|Transformers 4.37.2的LlamaModel在收集最终隐藏状态前已经调用最终norm，作者再对每个r调用norm，因此最后项被重复norm。中间项则调用一次。默认16–29不含最后项，但原选层以最后项作参照，会受此实现影响。先忠实保留作者实现，再单列差异诊断，不静默修正|
|词表|代码硬编码32000维。目标模型公开config同为32000，32层。加载时仍须断言lm_head输出维度和tokenizer实际大小，不将常量迁移到任意底座|
|拒绝词|20个字符串，各取encode(add_special_tokens=False)[0]，不是完整词序列，也不是搜索所有拼写。重复token索引只置1、不累加。实际token ID、拆分和碰撞须待tokenizer取得后保存，当前未假装已测|
|RTS来源|论文§3.1描述输出收集与逐层top5扩展，公开入口使用固定词表，未见该构建循环。首轮复用作者固定词表，不声称重做RTS发现过程|
|模板与位置|目标模型名命中llava_v1，user消息后追加空assistant，取完整模板最后位置。XSTest分支img=None，仍调用LLaVA模型，不转成纯Vicuna底座|
|投影与归一化|norm(r)后通过lm_head得到logits，取最后token的词表向量与稀疏拒绝向量做cosine。没有softmax，不是拒绝token概率和。当前投影所有token再取最后一项，不能先宣称已优化显存|
|积分及方向|np.trapz(F[s:e+1])，dx=1，保留带符号余弦。高分直接作为不安全正类的score传roc_curve，不取负号。空切片返回None，单点积分为0，后续须显式检查合法区间|
|标签|XSTest的type包含contrast才是toxicity=1，其余为0。不能使用final_label代替危险性，它是另一种回答标注|
|原选层|Funsafe均值−Fsafe均值，严格大于最后层差值的层号取最早s与最晚e；中间层即使未过线也包含。空集合没有论文给出的回退规则，应报告失败而非随意设默认值|
|指标|源码同时算AUROC和PR曲线梯形AUPRC，后者不应当作average_precision。论文主表AUROC，LLaVA/XSTest参考0.868，只作后续参照，不是本轮结果|

最终norm依据：[Transformers v4.37.2 Llama源码](https://raw.githubusercontent.com/huggingface/transformers/v4.37.2/src/transformers/models/llama/modeling_llama.py)，约948–982行。模型维度依据：[模型config](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/blob/main/config.json)。上述是静态行为推导，尚无真实模型张量验证。

## 数据、依赖和许可

仓库自带`data/few_shot.zip`为6,589,619字节。只读取压缩清单和JSON，未解压执行。其`few_shot/few_shot.json`含12条，正常6条、不安全6条，4条文本、8条图文，图像相对路径均以data/few_shot开头。运行原选层前须解压到正确工作目录，并校验与后续测试问题、图像重叠情况。校准集不能并入最终测试。

XSTest文件没有随仓库提供。默认路径为`data/xstest-v2-copy/data/gpt4-00000-of-00001.parquet`，loader也接受CSV。作者指向的[HF数据卡](https://huggingface.co/datasets/natolambert/xstest-v2-copy)显示gpt4 split共450行和CC-BY-4.0。后续下载须锁定revision、文件SHA256及行ID，并实际验证250/200数量，不能只信网页计数。当前未下载，未生成虚假样本清单。

作者要求Python 3.10，requirements固定torch2.1.2、transformers4.37.2、accelerate0.21.0、numpy1.26.4及大量CUDA12包，还含Linux NCCL与triton，不能直接当作当前Windows/Python3.12的可用安装方案。固定LLaVA版本的[pyproject](https://raw.githubusercontent.com/haotian-liu/LLaVA/c121f0432da27facab705978f83c4ada465e46fd/pyproject.toml)与主要模型依赖一致。parquet引擎未列入requirements，读parquet时需补并记录。peft0.14.0与旧transformers组合需在隔离环境做import核验，本轮未执行pip解析，不能宣称依赖已兼容。

锁定提交的完整树无LICENSE或COPYING文件，README也未声明独立代码许可。公开可读不等于可自由再分发。当前仅保存用户授权的本地研究副本，作者代码未修改，不对外发布。代码、few-shot图像、外部数据及模型分别核条款，不能沿用LLaVA的Apache声明覆盖HiddenDetect或权重许可。模型权重访问条件尚未完成核验。

## 拟定的最小冒烟

目标为学校Linux/4090，单底座LLaVA-v1.6-Vicuna-7B，batch=1、float16、无量化、作者llava_v1模板、固定20词表和默认0基闭区间[16,29]。先锁定XSTest gpt4 split，从正常／contrast各按ID稳定选4条，总8条，在运行前保存ID、prompt哈希、标签和来源版本。用于管线检查的8条不作为最终测试效果证据。

每条保存32维F、积分、标签、模板哈希、实际dtype/device和输入长度。核对无图分支、hidden_states长度33、词表32000、token ID、finite值与同一F重算积分一致。记录实际峰值显存和耗时。冒烟不计算可信性能、不调阈值，不与论文表格比较。默认区间原分数、作者12条多模态原选层和最终测试各自独立记录。

60GB是否足够尚未远程核查。7B FP16仅语言权重约14GB是参数估计，不是下载实测，还需CLIP、Conda/CUDA、缓存及临时文件。远程先查配额、驱动、Python/Conda和联网，指定一个缓存目录避免重复副本，再核模型revision、文件大小和条款后下载。模型卡的默认dtype与作者强制float16差异需记录。

## 阻塞与下一步

第一阶段本地审计已完成，远程环境核查待接入。学校SSH尚未配置，未获得可用连接别名和登录方式，不能开展远程命令、验证GPU或下载模型。下一步由用户在本机配置并确认可用SSH连接，再进行只读远程环境清点。保持主机地址、账户和凭据不进入网页和实验公开记录。
