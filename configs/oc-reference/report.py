"""Render the fixed OC-1 family, including negative results and evidence limits."""
import json
from core import ROOT,digest,write
OUT=ROOT/'results/oc-reference'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def ci(x):return '['+', '.join(f'{v:+.6f}' for v in x)+']'
def pct(x):return f'{x*100:.2f}%'
def main():
    new=load('confirmation-evaluation.json');hist=load('historical-evaluation.json');m=load('manifest.json')
    rules=json.loads((ROOT/'configs/oc-reference/rules.json').read_text());audit=load('token-audit.json')
    labels=['原20-token等权RV基线B','清理参照','扩充参照','原集合加权']
    lines=['# OC-1拒绝词参照：三个子实验验证与交接','',
    '日期：2026-10-30；运行ID：OC-RV1。现行OC-1指“拒绝词参照”，不是旧汇总实验的临时编号。用户批准的全部二级表见[流程实验集群](../../docs/oc-cluster-plan-2026-10-30.md)。','',
    '**本轮结论：保留原RV。** 清理版总体AUROC上升，但没有确认低误报排序改善，冻结阈值召回下降；本固定扩充版与加权版的主pAUC修正差区间全负。只否定这两套具体规则，不否定全部扩充或加权方法。','',
    '## 为什么做这轮、究竟改什么','',
    '原方法先把各层原输入末位隐藏状态做norm与词表投影，再与固定拒绝词向量求余弦，得到每层F，最后取[16,29]梯形积分。我们固定同底座、原模板末位、原norm与完整token投影形状、余弦、0基闭区间16至29、原梯形积分及高分危险方向；只改变参照集合或权重。无生成、无分类器、无重新选层，不同时改单层评分公式或跨层汇总。','',
    '三个子实验是评分前人工写定的语义假设，没有用训练/开发/测试标签筛词或调权重；但继承原few-shot选层背景，正常阈值校准也有正常标签，不能称全流程零标签。','',
    '|一级|改动位置|二级|大白话问题|本次固定改动|直接对照|','|---|---|---|---|---|---|',
    '|OC-1|拒绝词参照|OC-1-1|删掉不够明确的词，会更好吗？|原20字符串中保留固定10个，其他只删除|原20-token等权RV B|',
    '|OC-1|拒绝词参照|OC-1-2|增加拒绝表达，覆盖会更全吗？|原集合不删，新增10字符串，首token去重后29坐标|B|',
    '|OC-1|拒绝词参照|OC-1-3|拒绝语气比其他词重要吗？|原20坐标不变，固定5字符串对应坐标权重2，其他1|B|','',
    '- 原字符串：'+', '.join(rules['original'])+'。',
    '- 清理保留：'+', '.join(rules['clean_keep'])+'。这只是人工清理假设，不称被删词已被证明无关。',
    '- 扩充新增：'+', '.join(rules['expand_add'])+'。',
    '- 权重2：'+', '.join(rules['weight_two'])+'；其余原词权重1。2:1是固定工程取值，未搜索最优。',
    '- 每字符串仍只取第一个token，重复坐标取最大权重、不累加；完整编码、首token解码和碰撞信息均在token-audit。有效坐标数：'+str(audit['reference_support_counts'])+'。不对完整短语做额外前向。','',
    '## 数据、校准与冻结','',
    f"- 排除实际全部{m['historical_count']}条历史原文、{m['excluded_ids']}个含别名ID，包含旧pure-oc新2000和最新layer-selection新2000；检查历史清单原文哈希覆盖，规范化精确去重及字符3–5gram TF-IDF（65536特征）余弦≥0.92连通分组，历史优先。",
    '- 原计划目标新正常1000＋危险1000，但实际剩余仅正常300＋危险782。本批在任何参照变体评分前固定全部1082条，保留不平衡，没有用旧样本补足。这是同源剩余留出，不是官方新测试、跨领域或多模态确认；300正常仅给低FPR尾部有限精度，小改进检验功效有限。',
    '- 历史C1000和同400正常校准先评分；四参照分别取同校准集升序第381个，严格大于告警。不把不同参照的分数尺度混用，也不在新测试移阈值凑5%。',
    '- 60b15a0提交人工规则、token审计与清单；59fe852保留首锁。独立审查建议加强确认入口，3bbb897修复，70de042提交v2锁。历史阶段先验证原RV逐层F，再封印8份历史/校准/源码/清单证据，最后启动新1082；pre-confirmation-seal与每条新结果绑定。首锁保留，所有真实评分使用v2源码。',
    '- 首个源码包上传时连接重置，原端口后续恢复后重传成功；未换参数、词集合或数据。自动审批曾因额度故障拒绝一次远程检查，用户要求继续后重试通过。完整去重文件下载被审批以可能含私有原文为由拒绝，未绕过；完整审计留在学校，仅取无原文的计数/哈希摘要。','',
    '## 指标与判定规则','',
    '主指标：与既有OC协议一致的标准化pAUC(FPR≤0.1，sklearn McClish)。三个子实验各对B，共享标签分层配对bootstrap10000次、seed20261030，每项双侧98.333333%差区间控制三比较家族。AUROC和冻结阈值召回/误报差为补充95%区间；绝对率给Wilson95。所有区间以本次固定校准和规则为条件，不覆盖重采校准的不确定性。历史诊断不能声称新确认。','',
    '## 历史C诊断','',
    '|方法|同批B AUROC→候选|pAUC|相对B主pAUC差区间|召回/误报|','|---|---|---|---|---|']
    for j,v in enumerate(hist['metrics']):
        lines.append(f"|{labels[j]}|{hist['metrics'][0]['auroc']:.6f}→{v['auroc']:.6f}|{v['pauc']:.6f}|{ci(v['pauc_delta_ci983333']) if j else '基线'}|{pct(v['tpr'])}/{pct(v['fpr'])}|")
    lines+=['','清理版历史AUROC上升但pAUC下降，说明总排序和低误报尾部可能不一致；据此没有改词表或删去失败候选。','',
    '## 新1082条确认：全部三项','',
    '以下B在本批重新计算，不能拿C的0.746640当另一批箭头左侧。','',
    '|方向/方法|同批B AUROC→候选|pAUC|主pAUC差|修正98.333333%差区间|判定|','|---|---|---|---|---|---|']
    for j,v in enumerate(new['metrics']):
        verdict='共享基线' if j==0 else '低误报改善未确认' if v['decision']=='unconfirmed' else '本固定方案更差' if v['decision']=='worse' else '本轮确认改善'
        lines.append(f"|{v['direction_id']} / {labels[j]}|{new['metrics'][0]['auroc']:.6f}→{v['auroc']:.6f}|{v['pauc']:.6f}|{v.get('delta_pauc',0):+.6f}|{ci(v['pauc_delta_ci983333']) if j else '基线'}|{verdict}|")
    lines+=['','|方法|危险检出/782与召回|正常告警/300与误报|绝对召回Wilson95|绝对误报Wilson95|','|---|---|---|---|---|']
    for j,v in enumerate(new['metrics']):
        lines.append(f"|{labels[j]}|{v['tpr_count']}/782={pct(v['tpr'])}|{v['fpr_count']}/300={pct(v['fpr'])}|{ci(v['tpr_wilson95'])}|{ci(v['fpr_wilson95'])}|")
    lines+=['','上表Wilson区间单位为0～1的率。配对差补充：','',
    '|方向|AUROC差及95%区间|召回差95%|误报差95%|','|---|---|---|---|']
    for v in new['metrics'][1:]:lines.append(f"|{v['direction_id']}|{v['delta_auroc']:+.6f} {ci(v['auroc_delta_ci95'])}|{ci(v['tpr_delta_ci95'])}|{ci(v['fpr_delta_ci95'])}|")
    lines+=['','## 按问题闭环','',
    '- **OC-1-1 清理：** 推测泛词会带来噪声，只删固定10词；直接对照原RV。新同批AUROC0.756547→0.775682，但pAUC0.612375→0.603334，主差区间含0，改善未确认。危险检出246→214，正常告警18→16，不能把少报警描述为同误报召回改善。保留总体排序线索，不直接替换RV。',
    '- **OC-1-2 扩充：** 推测覆盖不足，只加词。新同批AUROC0.756547→0.705533，pAUC0.612375→0.572957，修正主差区间全负；危险检出246→179，正常告警同18。本固定扩充方案更差，保留失败项；不能推导所有扩充均无效。',
    '- **OC-1-3 加权：** 推测拒绝语气更重要，只改原支持坐标权重。新同批AUROC0.756547→0.740072，pAUC0.612375→0.575021，修正主差区间全负；危险检出246→173，正常告警18→20。本固定2:1规则更差，不据测试反调比例。',
    '- **下一步：** 维持原RV作为共享参照，后续若继续，优先在原集合上单独验证OC-2的单层评分方式；这需另行授权与预注册，不自动执行。若要重新筛词或学权重，应另记开发预算并取得新的确认数据；本批1082从此转历史，不能继续用作未见测试。','',
    '## 数值身份、成本和限制','',
    '- 历史C原AUROC精确0.746640，全部1400原RV逐层F与旧缓存exact；历史及新确认两阶段各16条探针F及积分差0，模型12文件SHA及作者源码版本/清洁状态通过。',
    '- 现有缓存只有最终层完整logits，未用它冒充全层。本轮重新前向，每层保留4个标量F、29个并集token的logits及全词表logit范数，供稀疏余弦重构；不保存完整词表缓存或完整隐藏向量。',
    '- 派生重构最初误用全float32归一化，最大差2.9745e-5；查明原混合精度余弦对logits先在fp16归一化，再与float32参照相乘。派生工具改按半精度范数/除法舍入重构，评分、阈值及封印方法均未改。相关回归测试先失败后通过，实际最大误差见verification。',
    '- 独立审查不访问原文、SSH或凭据，不能称它独立重跑私有文本去重或GPU。近重复排除降低重复风险，不保证语义独立；这里只验证一组人工清理、一组扩充、一组固定权重。']
    for phase in ['history','confirmation']:
        x=load('extraction-'+phase+'.json');lines.append(f"- {phase}提取{x['count']}条，{x['seconds']:.2f}秒（不含模型加载/文件哈希），峰值allocated {x['peak_allocated_bytes']/1024**3:.2f}GiB；三候选共享同一次模型前向，但增加了余弦计算，未证明模型推理加速。")
    lines+=['','## 证据与重放','',
    '- [固定规则](../../configs/oc-reference/rules.json)、[token完整编码与坐标审计](token-audit.json)、[样本清单](manifest.json)、[源码锁v2](source-lock-v2.json)、[首锁保留](source-lock.json)、[新评分前封印](pre-confirmation-seal.json)。',
    '- [历史逐样本F/投影小统计](history-scores.jsonl)、[新逐样本F/投影小统计](confirmation-scores.jsonl)、[阈值](thresholds.json)、[历史全统计](historical-evaluation.json)、[新确认全统计](confirmation-evaluation.json)、[无原文去重摘要](dedup-summary.json)。',
    '- [本地重放](verification.json)、[双端重放比较](cross-host-verification.json)、[独立审查](review-2026-10-30.md)、[历史身份](identity-history.json)、[确认身份](identity-confirmation.json)。',
    '- 当前实验目录运行 `python configs/oc-reference/test_contract.py`、`python configs/oc-reference/test_replay.py`、`python configs/oc-reference/verify.py` 可复算公开统计，不需要GPU或私有原文。完整重提取需授权环境、原数据和固定模型，须检出v2锁对应源码，不能把重放现有数据称为新确认。',
    f"- 新确认分数SHA256：`{digest(OUT/'confirmation-scores.jsonl')}`。",
    f"- 新评分前封印SHA256：`{digest(OUT/'pre-confirmation-seal.json')}`。"]
    (OUT/'report-2026-10-30.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('RV_CHINESE_REPORT_WRITTEN')
if __name__=='__main__':main()
