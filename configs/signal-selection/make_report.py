"""Render all frozen outcomes; never choose a winner from final tests."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'results/signal-selection'
freeze = json.loads((out / 'freeze.json').read_text())
evaluation = json.loads((out / 'evaluation.json').read_text())
search = json.loads((out / 'search.json').read_text())
lines = ['# 互补信号与 BPSO 前瞻实验：2026-10-28', '',
    '首轮实验已完成。互补隐藏状态信号在 Aegis 上具有增量信息，但本轮没有支持 BPSO 稳定优于等预算随机搜索的证据，跨来源召回也未保持。保留全部预声明种子与负面结果，不按最终测试挑选掩码。', '',
    '## 主要发现', '',
    '- Aegis：F32 监督探针 AUROC 0.83740，训练质心 C32 为 0.86487，融合 all64 为 0.87402；all64 相对 F32 的配对差值 95% bootstrap 区间为 +0.02116 至 +0.05138。这个区间以本次固定训练、选择和阈值为条件，不包含重训、数据选择或多重比较的不确定性。',
    '- 原固定积分 [16,29] 的 Aegis AUROC 为 0.77849。它没有使用新增的1600条监督训练标签，因此与监督探针的差异不能全部归于新信号；更合适的信号消融对照是 F32、C32、all64。',
    '- First-to-Know 风格的完整下一 token logits 探针是强对照：Aegis AUROC 0.87822，TPR 54.2%，FPR 5.8%；all64 为 0.87402、58.4%、7.6%。实际误报不同，不能把较高召回直接算作同等误报收益。两者 AUROC 配对差区间包含零。',
    '- BPSO 五种子 Aegis 平均 AUROC 0.87075，随机搜索为 0.87078，未见稳定提升。BPSO 选出24–30组，平均 TPR 57.84% / FPR 7.64%，完整64组为58.4% / 7.6%；部分种子的召回损失超过1个百分点。L1只含13个非零组，Aegis TPR59.6% / FPR7.6%，更简单的对照不能忽略。',
    '- JBB：all64 AUROC0.93130，F32同为0.93130；BPSO平均0.92830，TPR70.6%，低于all64的76.0%，FPR也由7.0%降到4.8%。因此不能声称稀疏方案跨来源无损。L1召回68.0% / FPR5.0%，也有代价。',
    '- ToxicChat是人类毒性标签的独立辅助任务。all64 TPR仅14.0% / FPR1.0%；logits探针为28.0% / 2.5%。这说明迁移阈值与任务差异值得研究，不能将此视作同等误报下的提升，更不能混入安全输入主任务平均分。', '',
    '## 数据、隔离与设置', '',
    '|角色|样本数|标签0 / 1|来源|', '|---|---:|---:|---|',
    '|训练|1600|800 / 800|Aegis官方train|', '|搜索开发集|600|300 / 300|Aegis官方train内独立组|',
    '|独立阈值校准|1000|500 / 500|Aegis官方validation|', '|主测试|1000|500 / 500|Aegis官方test|',
    '|外部行为测试|200|100 / 100|JBB benign/harmful|', '|辅助毒性测试|400|200 / 200|ToxicChat1123 test，human_annotation=true|', '',
    '数据 revision、文件哈希与许可见[下载锁](../../logs/signal-selection-download.json)和[来源配置](../../configs/signal-selection/data-sources.json)。原450条与作者12条优先排除；7535条候选经规范化精确重复和字符3–5gram TF-IDF余弦≥0.92的连通分组，共排除217条重复候选。只保留完整llava_v1模板不超过512 token的输入，不截断。此去重不保证语义独立；本轮结论限于这套短文本、英语为主且人工平衡的样本，不能由AP推断实际部署精确率。', '',
    'LLaVA-v1.6-Vicuna-7B、CLIP、FP16、作者模板均沿用锁定版本；不生成回答，不训练基础模型。F是32层作者拒绝向量余弦信号，保留最后一层重复norm；H是每层最后位置隐藏状态；C是训练集类质心的余弦差。C只是RCS启发的质心简化，原RCS投影和KCD尚未复现。完整logits探针是统一当前模型设置的First-to-Know风格适配，不能称原论文数值复现。', '',
    '探针标准化、质心和分类头只拟合训练集。除L1固定C网格外，逻辑回归C=1、balanced、liblinear、max_iter=1000；所有头收敛。选择目标为 dev AUROC − 0.02 × 组数/64。BPSO为20粒子×25轮，随机搜索同为每种子至多500个不同掩码。5个种子全部保留。连续区间496个为独立的固定积分对照；L1试4个C，贪心上限500，均未伪装成相同总计算成本。搜索全局复用相同掩码头；计数是各运行实际查询的不同掩码数，时间包含共享缓存影响。', '',
    '数据清单先以`472d8a9`提交；19个候选、训练头和质心以`3814579`提交后才执行校准／测试评估。freeze SHA256为`'+evaluation['freeze_sha256']+'`。每个方法只根据独立校准集500条安全输入取95%阶统计量，并严格使用 score > threshold；校准FPR均≤5%。最终测试FPR照实报告，不再移动阈值凑5%。', '',
    '## 全部冻结候选的测试结果', '',
    'TPR为危险／毒性样本召回，FPR为安全／非毒性样本误报。AUROC括号为500次按标签分层、方法间配对的prompt-group bootstrap 95%区间；各组本轮仅保留一条，所以组重采样等同于行重采样。TPR/FPR的bootstrap与Wilson区间、AP和混淆矩阵全部见[evaluation.json](evaluation.json)。区间不做多重比较校正，也不代表5种子重训的独立证据。', '']
for role, title in [('test_aegis', 'Aegis：1000条'), ('test_jbb', 'JBB：200条'), ('test_toxicchat', 'ToxicChat：400条，独立毒性任务')]:
    lines += ['### '+title, '', '|候选|组数|AUROC [95% CI]|AP|TPR|实际FPR|', '|---|---:|---:|---:|---:|---:|']
    for candidate in freeze['candidates']:
        name = candidate['name']; value = evaluation['tests'][role][name]
        lo, hi = value['ci95']['auroc']
        groups = '32000维logits' if candidate['groups'] is None else str(candidate['groups'])
        lines.append('|%s|%s|%.5f [%.5f, %.5f]|%.4f|%.1f%%|%.1f%%|' %
            (name, groups, value['auroc'], lo, hi, value['average_precision'], 100*value['tpr'], 100*value['fpr']))
    lines += ['', '|方法：五种子全部|平均AUROC ± 样本标准差|平均TPR|平均FPR|', '|---|---:|---:|---:|']
    for method in ('random', 'bpso'):
        values = [evaluation['tests'][role]['%s_seed%d' % (method, seed)] for seed in range(5)]
        lines.append('|%s|%.5f ± %.5f|%.2f%%|%.2f%%|' % (method,
            np.mean([v['auroc'] for v in values]), np.std([v['auroc'] for v in values], ddof=1),
            100*np.mean([v['tpr'] for v in values]), 100*np.mean([v['fpr'] for v in values])))
    lines.append('')
lines += ['## 预算、资源和重放', '', '|搜索|种子|不同候选数|秒|', '|---|---:|---:|---:|']
for run in search:
    lines.append('|%s|%s|%s|%s|' % (run['method'], run.get('seed', '—'),
        run.get('unique_evaluations', run.get('evaluations')), '%.3f' % run['seconds'] if 'seconds' in run else '未单独计时'))
lines += ['', 'GPU特征提取232.05秒，峰值allocated显存14,744,222,208字节（约13.73GiB），缓存1,566,105,984字节。训练／搜索85.15秒，最终评估37.96秒。训练和搜索单BLAS线程；基础模型未训练。首个8条训练输入的32层作者信号差值为0，积分一致。首次提取resumed_at=0，完整一次生成，无混合版本续跑。', '',
    '缓存续跑审查后增加了提取器、运行时、模型锁及LLaVA版本身份校验；本轮已有缓存保持原样，原提取实现可由提交`b38ba84`恢复，实际提取器SHA在[提取报告](../../logs/signal-selection-extraction.json)中。当前新版提取器会拒绝旧版缓存身份，不能将旧缓存元数据自动改为新版后冒称新提取；如需重新提取，应使用独立缓存目录或保留旧缓存后重新生成。训练和评估仍校验原清单及完整行边界。', '',
    '按源码重跑依次执行 `download_public.py`、`build_manifest.py`、`extract_features.py`、`train_search.py`、`evaluate_frozen.py`，都在仓库根目录使用固定env/bin/python。锁定清单与本次原始输入的records SHA应一致；既有freeze或evaluation存在时脚本拒绝覆盖，另建实验目录和新测试集才进行下一轮研究。本地无需模型即可用`verify_results.py`复核公开分数、混淆矩阵、点指标、头和清单哈希。数据、模型、H/logits大缓存不入Git；公开文件仅含索引、哈希、分数和训练头，不含原始提示文本。', '',
    '## 判断与后续边界', '',
    '本轮值得继续验证的是互补信号与跨来源阈值适配。BPSO在开发集的收益未稳定转化为测试收益，不能据此主张优化算法创新；L1与完整logits仍是必须保留的强基线。主任务部分压缩方案接近完整64组，但本轮未满足跨来源稳定、全种子无损并优于简单搜索的完整目标。', '',
    '组数减少只描述后处理所需信号；当前一次推理仍经过全部Transformer层，L1代码仍读取全64组后使用零系数，未实现或测量基础模型加速。未来若做新的实验，应先冻结跨来源训练／校准方案或原RCS实现，再取得未观察的新最终测试；本轮测试已经使用，不能重新当作未见最终测试。']
(out / 'report-2026-10-28.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(out / 'report-2026-10-28.md')
