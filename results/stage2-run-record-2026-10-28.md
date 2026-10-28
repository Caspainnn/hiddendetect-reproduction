# 第二阶段执行记录（2026-10-28）

本记录区分已发生的操作与尚未完成的推理，最终以对应结果JSON和日志为准。

## 已取得的材料

独立Python3.10环境已由Conda创建。现有base、qwen3vl_vllm、vllm环境未修改。远程复现目录位于用户家目录下的`HiddenDetect-reproduction`，VS Code Remote-SSH登录后可直接打开这个文件夹。环境目录为`env/`，权重在`models/`，脚本在`configs/`，来源、失败和运行日志在`logs/`。

作者仓库已在学校机器克隆，HEAD核对为`2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb`。本地复现工作另建Git仓库，分支为`reproduce/smoke`，忽略作者源码、权重、数据与环境目录，不覆盖作者仓库，也未推送GitHub。

## 模型来源与完整性

学校和本机无法直连Hugging Face，hf-mirror返回403。ModelScope可访问，因此使用`AI-ModelScope/llava-v1.6-vicuna-7b`及`AI-ModelScope/clip-vit-large-patch14-336`镜像。镜像文件revision、字节数与SHA256见[锁定清单](../configs/artifacts-lock.json)。三个LLaVA safetensors分片和CLIP pytorch_model.bin的SHA256均在下载前独立对照了HF官方文件页面：

- [LLaVA分片1](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/blob/main/model-00001-of-00003.safetensors)
- [LLaVA分片2](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/blob/main/model-00002-of-00003.safetensors)
- [LLaVA分片3](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/blob/main/model-00003-of-00003.safetensors)
- [CLIP权重](https://huggingface.co/openai/clip-vit-large-patch14-336/blob/main/pytorch_model.bin)

12个文件共15,838,464,773字节，下载后全部验证通过。实际下载及校验过程351.20秒，不能当作模型推理时间。证据：[完整性结果](../logs/artifacts-verified.json)、[下载日志](../logs/download-models.log)。仅保存一份模型文件，不再额外复制HF缓存。

[LLaVA模型卡](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/raw/main/README.md)声明Llama2 Community License并以研究为主要用途，本轮仅本地学术复现，不分发权重。CLIP的[上游代码许可](https://github.com/openai/CLIP/blob/main/LICENSE)为MIT，不能用它覆盖LLaVA或HiddenDetect的独立条款。HiddenDetect未发现独立许可，作者代码副本不纳入自己的可发布Git内容。

## 文本数据与样本固定

采用XSTest[原作者仓库](https://github.com/paul-rottger/xstest)，锁定`d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`的`xstest_prompts.csv`。实测450条，正常250条、不安全200条，`type`含contrast为正类，与作者loader规则一致。来源和Git blob记录见[xstest-source.json](../logs/xstest-source.json)，文件SHA256、样本ID与逐条prompt哈希见[冒烟清单](../configs/smoke-manifest.json)。

这与HiddenDetect默认指向的HF gpt4 parquet存在下载来源差异，尚未验证HF副本的450条内容是否完全一致。原作者CSV不含回答内容，本轮只读取prompt与type，不需要生成回答。最初使用了错误的CSV文件名，返回404，随后通过原仓库树核到正确文件名并成功取得，未将失败下载当作数据。

固定8条ID：正常`1、10、101、102`，不安全`100、126、127、128`。按字符串ID顺序每类取4条，不代表类别均匀或效果评价。与作者12条校准问题按忽略大小写、折叠空白后的精确文本比对，450条未发现重叠，不能据此声称语义上也无重叠。采样规则、8个ID及哈希在观测模型分数前锁定。

## 运行约束与验证

环境安装使用[作者推理依赖子集](../configs/requirements-inference-author.txt)，未安装网页UI或训练栈。核心torch、transformers、accelerate等保持作者指定版本，所有实际解析出的间接依赖在安装完成后另存freeze，不能称为作者完整环境逐项相同。

运行脚本直接导入并调用未修改的作者`test(samples, model_path, s=16, e=29)`。通过只读观察钩子记录前向输入长度、token哈希和32层余弦信号，保持norm、lm_head、词表首token和np.trapz不变。CLIP通过工作目录下`openai/clip-vit-large-patch14-336`本地路径别名加载相同权重，开启离线模式，防止运行时再次联网下载。

无模型单元检查已完成：下载文件损坏／截断拒绝，采样稳定与类别平衡、校准排除、重复ID拒绝，闭区间积分与32层计数校验。都先观察到缺实现失败，再实现并通过。本地通过不等于GPU推理通过。

GPU运行前须验证依赖导入、CUDA及模型加载，运行后须验证8条的32维F均为有限值，重算[16,29]梯形积分与作者输出一致。冒烟不报告有意义的AUROC，不调阈值，不能称多模态复现。

## 当前停点

权重与数据已准备。依赖安装进行中，尚无模型推理结果。后续运行结果和错误将追加到本记录及progress.md，不以已下载模型代替冒烟通过。
