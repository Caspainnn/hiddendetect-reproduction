"""Explicitly separate historical diagnostics from the earlier prospective OC-3 evidence."""
import json
from core import ROOT,digest
OUT=ROOT/'results/oc23'
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def ci(v):return '['+', '.join(f'{x:+.6f}' for x in v)+']'
def pct(v):return f'{100*v:.2f}%'
def main():
    cohorts=[load('historical-C-evaluation.json'),load('historical-RV-evaluation.json')]
    m=load('manifest.json');assert m['confirmation_enabled'] is False
    names=['原F＋原积分B','拒绝logit均值','拒绝logit平滑最大','拒绝概率质量','原F等权均值','原F最强4层','原F截尾均值','原F连续4层窗口','原F后7层减前7层']
    lines=['# OC-2单层评分与OC-3跨层汇总：补全、历史验证及交接','',
    '日期2026-10-30；运行OC-SA1。用户授权继续OC-2、OC-3。**本轮完成三项OC-2与新增OC-3-5的历史诊断，并重放OC-3已有四项；没有新增独立确认。**','',
    '## 为什么没有新确认','',
    f"实际核查{m['historical_count']}条历史原文、{m['excluded_ids']}个含别名ID，覆盖原EI、汇总OC、选层和最新OC-RV1等全部批次；从固定版本Aegis train/validation/test三分区再排除历史ID、原文规范化及近重复后，仅剩正常10、危险212。预先约定每类至少30才启动新确认，故confirmation_enabled=false，实际没有评分这些剩余样本，没有用旧数据充新。",
    '', '当前2482条全部已观测历史：C1000、400正常校准、OC-RV1的1082（300正常/782危险）。下面新评分算法即使出现正差区间，也只能称历史线索，不能称本轮独立确认提升。若需要继续前瞻验证，必须取得新来源/新批次并重新登记排除与协议。','',
    '## 固定边界与具体加工','',
    '固定模型、原模板输入末位、norm、原完整token词表投影形状、原20个首token参照、0基闭区间[16,29]和高分危险方向。无生成、无分类器或层重选；原few-shot及400正常校准仍有标签背景。OC-2只替换每层评分再原积分，OC-3只替换原F的跨层汇总，不组合两处改动。','',
    '|一级|二级|大白话问题|本次固定公式|','|---|---|---|---|',
    '|OC-2|OC-2-1|直接看拒绝token的平均分，更有用吗？|每层20个拒绝logits的float32均值，再原14层梯形积分|',
    '|OC-2|OC-2-2|某些拒绝表达很强，是否更值得重视？|每层tau=1的log-mean-exp：logsumexp(z_R)−log(20)，再原积分|',
    '|OC-2|OC-2-3|拒绝词在整个词表占的概率，更有用吗？|每层exp(logsumexp(z_R)−logsumexp(z_all))，float32稳定概率质量，再原积分|',
    '|OC-3|OC-3-1|端点为什么半权？|原14F等权平均；已完成方法的补充重放|',
    '|OC-3|OC-3-2|强信号被其他层稀释了吗？|原14F中最大的4个均值；已有方法|',
    '|OC-3|OC-3-3|极端高低分是否干扰判断？|原14F两端各去2，余10平均；已有方法|',
    '|OC-3|OC-3-4|连续高分是否更可靠？|11个连续4层窗口均值取最大；已有方法|',
    '|OC-3|OC-3-5|更深层相对早层上升，是否有用？|mean(F23…29)−mean(F16…22)，固定后7减前7；不翻方向、不取绝对值|','',
    '共同B是原20-token余弦F的梯形积分。OC-2没有沿用OC-1的清理、扩充或加权参照，tau=1事先固定不搜索。概率质量使用完整词表分母，未用L2范数代替。初始模型信息相同，但不同压缩算子可能保留不同部分，这正是本轮OC研究范围。','',
    '## 冻结与统计规则','',
    '- 源码/清单协议56aec3e，执行锁2df90fb，在变体提取前提交；无新确认状态也同时冻结。全模型文件哈希、作者源码版本/清洁状态及16旧探针先核验，再处理2482历史输入。',
    '- 同400历史正常校准，每个方法分别取升序第381个阈值，严格大于告警；不同分数尺度不混用阈值，不在评估集调到5%。',
    '- 八候选各对B，一个主家族；标签分层共享配对bootstrap10000次、seed20261030；各双侧99.375%主pAUC差区间（八比较Bonferroni）。pAUC为FPR≤0.1的sklearn标准化值。AUROC/召回/误报差95%与绝对率Wilson95均补充。',
    '- 两批都是历史诊断，不因区间全正升级为独立确认。机制比较top4对等权、截尾对等权、窗口对top4只作描述性95%，不挪用主家族显著性。所有区间以固定规则和校准为条件，不覆盖校准抽样不确定性。','']
    for e in cohorts:
        label='历史C1000（500正常/500危险）' if e['batch']=='historical_C' else '历史OC-RV1 1082（300正常/782危险）'
        lines+=['## '+label,'','|方法|同批B AUROC→候选|pAUC|主pAUC差|修正99.375%差区间|召回/误报|历史解释|','|---|---|---|---|---|---|---|']
        b=e['metrics'][0]
        for j,v in enumerate(e['metrics']):
            result='基线' if j==0 else '正向诊断线索，未独立确认' if v['decision']=='positive_interval' else '本批更差' if v['decision']=='worse' else '改善未获区间支持'
            lines.append(f"|{v['direction_id']} {names[j]}|{b['auroc']:.6f}→{v['auroc']:.6f}|{v['pauc']:.6f}|{v.get('delta_pauc',0):+.6f}|{ci(v['pauc_delta_ci99375']) if j else '基线'}|{pct(v['tpr'])}/{pct(v['fpr'])}|{result}|")
        lines+=['','|方法|AUROC差95%|召回差95%|误报差95%|绝对召回Wilson95|绝对误报Wilson95|','|---|---|---|---|---|---|']
        for j,v in enumerate(e['metrics']):
            lines.append(f"|{v['direction_id']}|{ci(v['auroc_delta_ci95']) if j else '基线'}|{ci(v['tpr_delta_ci95']) if j else '基线'}|{ci(v['fpr_delta_ci95']) if j else '基线'}|{ci(v['tpr_wilson95'])}|{ci(v['fpr_wilson95'])}|")
        lines+=['','以上区间为0～1率的差或率，不是百分数。机制描述：']
        for v in e['mechanisms_descriptive']:lines.append(f"- {v['comparison']}：pAUC差{v['pauc_delta']:+.6f}，描述性95% {ci(v['ci95_descriptive'])}。")
        lines+=['']
    lines+=['## 逐问题结论：只依据真实同批对照','',
    '|方向|固定问题/改动|历史两批的主pAUC差|解释边界|','|---|---|---|---|']
    for j in range(1,9):
        a,b=cohorts[0]['metrics'][j],cohorts[1]['metrics'][j]
        interpretation='两批主差区间全正，保留历史线索，仍缺独立确认' if a['decision']==b['decision']=='positive_interval' else '两批主差区间全负，本固定方案不支持升级' if a['decision']==b['decision']=='worse' else '结果未形成一致区间支持，不据最高点挑选'
        lines.append(f"|{a['direction_id']}|{names[j]}|C {a['delta_pauc']:+.6f}；RV {b['delta_pauc']:+.6f}|{interpretation}|")
    lines+=['','OC-3-5只测试预注册的后7减前7，负向或差表现不能通过看过结果后翻符号补救，也不否定所有层间关系加工。OC-2三项只测试固定原词集合与原积分下的这些定义，不扩展到换词、温度搜索或训练检测头。','',
    '## OC-3已有独立确认：保留原证据，不算本轮新实验','',
    '运行OC-A1-N曾在另2000条同源留出上先冻结后评分。该批如今已是历史；下面保留当时四比较家族98.75%区间，不用本轮八比较99.375%去追改旧结论，也不把不同批次AUROC串成提升曲线。','',
    '|现行编号|冻结旧编号|该批B AUROC→候选|pAUC|原主差98.75%区间|当时结论|','|---|---|---|---|---|---|']
    prior=json.loads((ROOT/'results/pure-oc/confirmation.json').read_text())
    for new,v in zip(['OC-3-1','OC-3-2','OC-3-3','OC-3-4'],prior['metrics'][1:]):
        lines.append(f"|{new}|{v['direction_id']}|{prior['metrics'][0]['auroc']:.6f}→{v['auroc']:.6f}|{v['pauc']:.6f}|{ci(v['pauc_ci9875'])}|{'小幅同源排序改善确认' if v['decision']=='confirmed_improvement' else '主收益未确认'}|")
    lines+=['','## 资源、重放与限制','',
    '- 每条输入只有一次模型前向，共享原末位各层词表结果。已有小缓存没有全词表softmax分母，故重新提取原F、20拒绝logits、全词表logsumexp、L2范数及三项层分数；不保存完整隐藏向量或完整词表大缓存。',
    '- 本地与服务器核对原F/积分、三单层评分充分统计、既有四汇总逐样本分数及两批完整10000-bootstrap；原F的混合精度余弦按fp16归一化重构，生产三评分为float32，独立float64重构按记录容差核验，积分与统计直接精确重放。独立审查未访问私有原文、SSH或凭据，不能声称独立重跑GPU或全部原文去重。']
    verification=load('local-verification.json')
    lines.append(f"- 两批共{sum(verification['numeric_values_checked'].values())}个统计数值双端重放通过（容差1e-12）；原F及九列汇总逐项精确。三项单层float64重构最大绝对差分别为{verification['max_layer_score_reconstruction_differences']}，原混合精度余弦重构最大差{verification['max_original_cosine_reconstruction_difference']:.3g}。这些是派生精度核查，不改变已冻结生产分数。")
    ex=load('extraction-history.json');lines.append(f"- 2482条提取{ex['seconds']:.2f}秒（不含模型加载/模型哈希/身份探针），峰值allocated {ex['peak_allocated_bytes']/1024**3:.2f}GiB。未证明基础模型推理加速。")
    lines+=['','## 交接与下一步','',
    'OC-2三个子实验、OC-3-5的实现/历史诊断已补齐，OC-3-1至4的旧独立证据保留并完成补充重放。**独立确认没有完成，原因是可用未见正常样本只有10条，不是“验证通过”。** 后续先解决新来源/新批次及校准隔离，再对固定候选做一次前瞻确认；本轮不自动启动OC-4，也不继续在历史分数上搜索温度、层差方向、窗口或k。','',
    '## 证据与复现','',
    '- [冻结协议](../../docs/superpowers/plans/2026-10-30-oc23.md)、[全历史/剩余数量清单](manifest.json)、[原token审计](token-audit.json)、[来源锁](source-lock.json)、[无原文去重摘要](dedup-summary.json)、[未执行确认说明](confirmation-unavailable.json)。',
    '- [逐样本充分统计与分数](history-scores.jsonl)、[校准阈值](thresholds.json)、[历史C完整统计](historical-C-evaluation.json)、[历史RV完整统计](historical-RV-evaluation.json)、[身份](identity-history.json)、[资源](extraction-history.json)。',
    '- [本地验证](local-verification.json)、[服务器验证](verification.json)、[双端比较](cross-host-verification.json)、[独立审查](review-2026-10-30.md)。',
    '- 实验仓库目录运行 `python configs/oc23/test_contract.py` 和 `python configs/oc23/verify.py` 可重放公开统计，不需要GPU或私有原文；完整模型重提取依赖已授权环境和来源锁对应源码，不得重跑历史输入后称为新确认。',
    f"- 历史充分统计SHA256：`{digest(OUT/'history-scores.jsonl')}`。"]
    (OUT/'report-2026-10-30.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('OC23_REPORT_WRITTEN')
if __name__=='__main__':main()
