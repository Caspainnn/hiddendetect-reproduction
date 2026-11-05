"""Produce the plain-language report from verified, matched EI-V1 results."""
import json
from core import OUT,digest

def main():
 e=json.loads((OUT/'evaluation.json').read_text());v=json.loads((OUT/'verification.json').read_text());search=json.loads((OUT/'search.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());seal=json.loads((OUT/'pre-evaluation-seal.json').read_text())
 assert v['passed'] and v['evaluation_sha256']==digest(OUT/'evaluation.json')
 names=['B-INT','B-F']+['EI-2-'+str(i) for i in range(1,7)]+['Fcopy']+['Noise-'+str(i) for i in range(1,5)]
 titles={'B-INT':'原14F梯形积分','B-F':'原14F＋匹配逻辑回归','EI-2-1':'F＋拒绝词逐坐标','EI-2-2':'F＋非拒绝词坐标','EI-2-3':'F＋残差PCA','EI-2-4':'F＋logits幅度','EI-2-5':'F＋完整归一化词表','EI-2-6':'F＋完整原始logits','Fcopy':'只复制F，不增加信息'}
 lines=['# EI-V1：原F加词表信息的匹配验证','',
 '2026-11-04启动，2026-11-05恢复连接后完成独立复核。文件日期沿用运行开始日期。','',
 '这次最清楚的线索是：补充参照外的少量词坐标（EI-2-2）、补充残差PCA（EI-2-3），在三个历史批次相对匹配F基线的AUROC差区间都为正。完整归一化词表（EI-2-5）也在三个批次有正差，但训练明显更贵；保留原始幅度（EI-2-6）相对EI-2-5的额外收益在三批都未获区间支持。只补幅度（EI-2-4）的效果随来源变化。','',
  '按事先约定的dev总体AUROC均值，完整归一化词表EI-2-5为0.913300，是完整端点中优先跟进的候选；完整原始logits EI-2-6为0.912367。若考虑更少输入和更低训练成本，首轮EI-2-2为0.895322，优先于其他首轮方案。这里只提出后续未见来源确认的候选，不据历史测试挑部署冠军。','',
 '本次完成六种“原14F＋补充信息”方案、匹配F基线和五种诊断控制，共108次拟合。旧OC-4-3／迁入EI，对应现行EI-2-1～6；当前OC-4-3仍指树模型。旧历史完整logits探针和OC-4成绩均未挪用。','',
 '所有测试都曾用于历史实验，本次是历史数据上的匹配验证，不是独立新确认，也没有据测试表现改特征、调参或挑种子。总体AUROC为主；下面统一列预先固定的首种子20261104，其他两种子保留在机器结果和后文。','',
 '## 这次具体改了什么','',
 '原模型、模板末位、0基16～29层和原F固定。检测头一律L2逻辑回归liblinear，C为0.1/1/10，每臂三候选×三种子，按dev总体AUROC均值选C；train1600（800/类）、dev600（300/类）、safe-cal400，全臂完全相同，不合并dev重训。坐标选择、PCA和标准化均只拟合训练集。','',
 'EI-2-1保留原拒绝词的单独坐标；EI-2-2每层取训练方差最大的32个非拒绝坐标；EI-2-3每层保留拒绝方向之外的4个PCA分量；EI-2-4补充每层logits范数的对数；EI-2-5/6分别保留14层全部归一化/原始词表坐标。所有候选都明确保留原14F。','',
 '原F逐值沿用原norm、完整序列GEMM和半精度余弦；补充词表从原float16 logits转float32，用float64范数再转float32。概念上的u·r̂与原F有舍入差异，不宣称逐位相同；原F通道没有用该近似值替换。','',
 '## 同批结果','',
 'pAUC采用FPR≤0.1的标准化定义。工作点由旧来源safe400第381顺序统计量确定，严格大于阈值告警，各候选各自校准。表内误报分母为安全输入数。AUROC或pAUC提高，不代表该固定阈值下的误报一定下降。','']
 for role,label,n0,n1 in [('C','历史C1000',500,500),('RV','历史RV1082',300,782),('OR_test','历史OR-Bench1500',1000,500)]:
  c=e['cohorts'][role];lines+=['### '+label+'：'+str(n0)+'正常／'+str(n1)+'危险','',
   '|方案|AUROC|低误报pAUC|检出率|误报率|误报数|','|---|---:|---:|---:|---:|---:|']
  for name in names:
   m=c[name][0];lines.append(f"|{name} {titles.get(name,'同维随机噪声控制')}|{m['auroc']:.6f}|{m['pauc']:.6f}|{m['tpr']:.2%}|{m['fpr']:.2%}|{m['fp']}/{n0}|")
  lines+=['','AUROC配对差（不是跨批相减）：','', '|比较|AUROC差|区间|判断|','|---|---:|---|---|']
  for name,comparison in c['primary_auroc_differences'].items():
   lo,hi=comparison['ci9875'];decision='本批差区间为正' if lo>0 else ('本批差区间为负' if hi<0 else '差区间含零')
   lines.append(f"|{name} − B-F|{comparison['delta']:+.6f}|98.75% [{lo:+.6f}, {hi:+.6f}]|{decision}|")
  for name,comparison in c['full_auroc_differences'].items():
   lo,hi=comparison['ci975'];decision='本批差区间为正' if lo>0 else ('本批差区间为负' if hi<0 else '差区间含零')
   lines.append(f"|{name}|{comparison['delta']:+.6f}|97.5% [{lo:+.6f}, {hi:+.6f}]|{decision}|")
  lines+=['']
 lines+=['## OR来源重新校准后的工作点','',
  '仍使用相同模型，只用另一组已使用过的OR-safe400定阈值。这里改变的是校准来源，不是模型重新训练。','',
  '|方案|检出率|误报率|误报数|','|---|---:|---:|---:|']
 for name in names:
  m=e['cohorts']['OR_test'][name][0]['source_calibrated'];lines.append(f"|{name}|{m['tpr']:.2%}|{m['fpr']:.2%}|{m['fp']}/1000|")
 lines+=['','## 容量、调参与成本','',
  '补零到512只是接口相同；不是相同有效容量。F复制仍只由原14F决定，额外数值秩可受浮点标准化舍入影响，不能称新增语义信息。完整词表未筛选或压缩，两个端点都使用448014维，并沿用同一求解器和相同调参次数。','',
  '|方案|原始维数|接口维数|可变列|训练数值秩|含截距参数数|所选C|9次拟合秒数|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,d in e['dimensions'].items():
  cap=v['capacity'][name];seconds=sum(r['seconds'] for r in search if r['name']==name)
  lines.append(f"|{name}|{d['raw_dimension']}|{d['interface_dimension']}|{d['nonconstant_columns']}|{cap['train_rank']}|{cap['weight_count_including_intercept']}|{seal['selected_C'][name]}|{seconds:.2f}|")
 lines+=['',
  '首轮秩用float64 SVD计算（输入为实际float32设计矩阵）；完整端点用float64训练Gram矩阵特征值和记录的容差计算，二者数值门槛不同，不把微小秩差当作信息增益。详见verification.json。','',
  '拟合秒数只含LogisticRegression.fit，不含模型加载、GPU提取、PCA、标准化、矩阵秩和bootstrap。此次没有对这些阶段分别计时，不能用拟合秒数当端到端成本。原logits缓存约5.49 GiB；维数与矩阵秩也不能代替推理成本测量。','',
  '## 三种子的AUROC波动','', '|批次|方案|最小AUROC|最大AUROC|','|---|---|---:|---:|']
 for role in ['C','RV','OR_test']:
  for name in names:
   values=[x['auroc'] for x in e['cohorts'][role][name]];lines.append(f'|{role}|{name}|{min(values):.6f}|{max(values):.6f}|')
 lines+=['','## 能得出什么结论','',
  '逐项结论以同批差区间为准，不能把原积分到监督头的变化归因于更多信息。EI-2-1～4主要对B-F；EI-2-5对B-F；EI-2-6主要对EI-2-5。不能把EI-2-6相对F的改善直接说成“幅度比归一化方向更好”。','',
  '噪声和F复制只作描述性诊断，不在看成绩后改成新的确认性比较。PCA失败只说明这次4维压缩方案不理想；方差选词失败也不证明参照外没有信息。所有方案同时改变可用信息、自由方向及正则化几何，不能把收益全解释为纯信息增益。','',
  '每个历史批次分别共享10000次分层配对重采样。首轮四比较各98.75%；完整端点两比较各97.5%，这是两个分别定义的家族，不是所有方案和所有数据集的联合95%保证。区间只覆盖固定已训练模型在该历史批次的样本不确定性，不覆盖训练/选型/校准，也不消除历史反复观察的影响。','',
  '若要形成部署结论，仍需在未用来源上冻结方案确认；优先级只能按预定dev标准提出，不按这些历史测试挑一个“冠军”再声称泛化最佳。','',
  '## 复核与归档','',
  f"共{len(manifest['samples'])}条；原F全量与既有缓存差最大{max(v['all_F_cache_max_differences'].values()):.3g}；重载头与变换重算分数差最大{v['reloaded_prediction_max_difference']:.3g}。AUROC、pAUC、旧阈值工作点和OR重新校准工作点均独立复算。",'',
  f"首次复核因把PCA整批矩阵计算改为128行分块而触发1e-7得分容差；恢复相同四线程、相同整批PCA后与冻结分数完全一致。未放宽容差、未修改特征/模型/归档分数；仍以分块重载头核验。PCA分块与整批特征差最大{v['pca_chunk_vs_frozen_cohort_max_difference']:.3g}，细节见pca-batching-diagnostic.json。",'',
  f"执行源封印见source-lock.json；核验提交{v['verification_commit']}。evaluation.json SHA256：{v['evaluation_sha256']}。",'',
  '原始提示词、模型、原logits、PCA和头权重留在服务器私有cache；本地只同步样本ID/哈希、小分数、协议和报告。首次模型加载失败未发生训练/评分，旧封印与错误日志保留于attempt-1，不改写旧冻结记录。','']
 (OUT/'report-2026-11-04.md').write_text('\n'.join(lines),encoding='utf-8');print('REPORT_WRITTEN')

if __name__=='__main__':main()
