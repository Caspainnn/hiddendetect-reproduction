"""Publish only verified metrics; derive descriptive subgroups without selecting."""
from pathlib import Path
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/ei2-images';ARMS=[f'EI-2-8-{i}' for i in range(1,6)];BASE='B-MM'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 from sklearn.metrics import roc_auc_score
 proof=json.loads((OUT/'verification.json').read_text());assert proof['passed'] and proof['selected_entries_bound']==42
 coverage=json.loads((OUT/'input-coverage-audit.json').read_text());assert coverage['passed'] and coverage['rows_checked']==1270 and coverage['evaluation_sha256']==proof['evaluation_sha256']
 for name,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256'),('source-lock.json','source_lock_sha256')]:assert sha(OUT/name)==proof[key]
 e=json.loads((OUT/'evaluation.json').read_text());m=json.loads((OUT/'manifest.json').read_text());rows=m['samples'];scores=np.load(OUT/'scores.npz');ex=json.loads((OUT/'extraction.json').read_text());recovery=(OUT/'recovery-provenance.json').exists();expected={'main_multimodal':1270,'neutral_image':1270,'caption_generation':1270,'perturbed_image':2540};cost_complete=not recovery and all(ex['forward_counts'].get(k)==v for k,v in expected.items())
 if recovery:cost_complete=bool(ex.get('cost_complete',False))
 cost={'passed':True,'evaluation_sha256':proof['evaluation_sha256'],'counters_complete_for_registered_operations':cost_complete,'recovery_used':recovery,'mandatory_single_process_counts':expected,'observed_counts':ex['forward_counts'],'timing_scope':'Registered GPU/forward operations only; loading, some CPU/I/O and OCR are not all included. This is joint extraction cost, not per-arm deployment latency.','lower_bound_if_incomplete':not cost_complete};(OUT/'cost-audit.json').write_bytes((json.dumps(cost,indent=2)+'\n').encode())
 subgroups=[]
 for role in ['test','task_test']:
  rr=[r for r in rows if r['role']==role];yy=scores[role+'__y']
  for category in sorted({r['category'] for r in rr}):
   ix=np.array([i for i,r in enumerate(rr) if r['category']==category]);y=yy[ix]
   for name in [BASE]+ARMS:
    s=scores[role+'__'+name][0,ix];cut=e['thresholds'][name][0];both=len(np.unique(y))==2;subgroups.append({'role':role,'category':category,'name':name,'n':len(ix),'clusters':len({rr[int(i)]['group_id'] for i in ix}),'auroc':float(roc_auc_score(y,s)) if both else None,'pauc':float(roc_auc_score(y,s,max_fpr=.1)) if both else None,'tpr':float((s[y==1]>cut).mean()) if np.any(y==1) else None,'fpr':float((s[y==0]>cut).mean()) if np.any(y==0) else None,'fp':int((s[y==0]>cut).sum()),'n_safe':int((y==0).sum())})
 (OUT/'subgroups.json').write_bytes((json.dumps({'descriptive_only':True,'evaluation_sha256':proof['evaluation_sha256'],'records':subgroups},indent=2)+'\n').encode())
 geometry=[]
 for role in ['train','dev','test','task_test']:
  scenes={}
  for row in rows:
   if row['role']==role:scenes.setdefault(row['scene_id'],{})[row['label']]=(row['width'],row['height'])
  pairs=[value for value in scenes.values() if set(value)=={0,1}];geometry.append({'role':role,'scene_pairs':len(pairs),'same_resolution_pairs':sum(value[0]==value[1] for value in pairs),'same_resolution_fraction':float(np.mean([value[0]==value[1] for value in pairs]))})
 (OUT/'image-geometry-audit.json').write_bytes((json.dumps({'descriptive_only':True,'evaluation_sha256':proof['evaluation_sha256'],'records':geometry,'not_a_causal_control':'Dimensions are not explicitly given to heads, but image preprocessing can expose geometry/source shortcuts; this audit does not rule them out'},indent=2)+'\n').encode())
 labels={BASE:'仅图文14F','Noise14':'14F＋同维噪声','B-T':'仅无图14F','OCR-only':'14F＋仅OCR分支','Caption-only':'14F＋仅描述分支','Neutral-description':'14F＋等长中性描述','ROI-uniform':'14F＋不加权视觉区域','ROI-random':'14F＋随机加权视觉区域','Neutral-image-delta':'14F＋中性图差异'};titles=['图片整体特征','图片转文字分支','有图无图差异','问题相关局部区域','图片扰动敏感度'];labels.update(dict(zip(ARMS,titles)))
 lines=['# EI-IM1：五种图片补充信息的正式图文验证','',
'本轮使用真实图片输入，保留同批原14F_mm和原L2检测头，只增加14个特征。五项及七个诊断臂全部完成126次LR拟合，并完成42最终头绑定、全部指标与按关联组区间复核。','',
'**任务边界：这是MSSBench情境安全介入检测。** 1类表示图片情境需要安全提醒/介入，不等于用户恶意、越狱成功或应该一律拒答。仅一种数据来源与一个基础模型，不能用这些分数与旧C/RV/OR纯文本批次相减。','',
'## 冻结数据与公平对照','',
'公开官方MSSBench revision `12396d24157427b671d97f23a2e370fca98fdf55`，752张原图逐一核验LFS哈希。相同/近重复图片和原始意图联合分组，同组不跨角色。统一服务器JPEG解码后重建像素记录，分组、样本ID和角色保持完全不变；训练前差异记录保留。VLGuard因官方gated且无缓存，本轮未使用或绕过访问控制。','',
'|角色|正常|需介入|独立关联组|','|---|---|---|---|','|train|320|320|140|','|dev|86|86|36|','|cal|82|0|36|','|聊天test|112|112|53|','|跨任务task_test|76|76|4|','',
'跨任务只有4组，不能按152条独立样本计算不确定性，也不冒称外部新来源。官方参考图片描述、图像路径、标签与类别不交给检测头。','',
'原0基16～29层、llava_v1模板末位、真实anyres视觉token，原norm/词表投影/拒绝词余弦。末位投影优化预检误差超过1e-7，因此全轮保留原完整序列投影；标准images前向与显式视觉嵌入前向F差为0。注意力只读最后一行，经模型直接输出校验后使用，不改变原后端。','',
'主臂B-MM、Noise14、五候选各3C×3种子，共63拟合；七个诊断臂再63拟合，总126。仅用640条train拟合PCA、质心、标准化与头；172条dev平均AUROC选C，精确平局按网格顺序。选择封印后才提取cal/test。三种子20261104/05/06全保留，主展示预定首种子。','',
'|方案|最终C|平均dev AUROC|','|---|---|---|']
 for name in [BASE]+ARMS:lines.append(f"|{name} {labels[name]}|{e['selected'][name]['C']}|{e['selected'][name]['dev_auroc_mean']:.6f}|")
 decisions=[]
 for role,title in [('test','聊天留出：224条、53关联组（主检验）'),('task_test','同源跨任务留出：152条、仅4关联组（描述性）')]:
  d=e['cohorts'][role];lines+=['','## '+title,'']
  if role=='task_test':lines+=['使用官方canonical safe_instruction/unsafe_instruction；两种情境的文字可能不同，因此同时列B-T，不能把跨任务变化全归因于图片。','']
  lines+=['|方案|AUROC|低误报pAUC|检出率|误报|','|---|---|---|---|---|']
  for name in [BASE]+ARMS+['Noise14','B-T','OCR-only','Caption-only','Neutral-description','ROI-uniform','ROI-random','Neutral-image-delta']:
   s=d[name][0];values=[]
   for key in ['auroc','pauc','tpr','fpr']:
    value=f"{s[key]:.6f}" if key in ['auroc','pauc'] else f"{s[key]:.2%}";base=d[BASE][0][key];improved=s[key]<base if key=='fpr' else s[key]>base
    if name in ARMS and improved:value='**'+value+'**'
    if key=='fpr':value+=f"（{s['fp']}/{s['n_safe']}）"
    values.append(value)
   lines.append('|'+name+' '+labels[name]+'|'+'|'.join(values)+'|')
  lines+=['','|候选相对同批B-MM|AUROC变化|差异区间|解释|','|---|---|---|---|']
  for name in ARMS:
   c=d['comparisons'][name];lo,hi=c['ci'];before=d[BASE][0]['auroc'];after=d[name][0]['auroc'];display=f'**{after:.6f}**' if after>before else f'{after:.6f}';decision=('改善获主区间支持' if lo>0 else ('下降获主区间支持' if hi<0 else '差异未确认')) if role=='test' else '仅描述性，4组不支持广泛泛化结论';lines.append(f"|{name}|{before:.6f} → {display}|{c['confidence']:.0%} [{lo:+.6f}, {hi:+.6f}]|{decision}|")
   if role=='test':decisions.append(name+'：'+decision)
  lines+=['','10000次按原图/意图关联组有放回配对bootstrap。主test五比较Bonferroni，各99%；task_test为未校正95%描述性区间。它们以当前训练/选择为条件，不包含重新采样训练标签的不确定性。辅助臂不进入主显著性判定。','', '|三种子范围|AUROC|pAUC|检出率|误报率|','|---|---|---|---|---|']
  for name in [BASE]+ARMS:
   values=[]
   for key in ['auroc','pauc','tpr','fpr']:
    lo,hi=min(s[key] for s in d[name]),max(s[key] for s in d[name]);values.append(f'{lo:.6f}～{hi:.6f}' if key in ['auroc','pauc'] else f'{lo:.2%}～{hi:.2%}')
   lines.append('|'+name+'|'+'|'.join(values)+'|')
 lines+=['','## 结论与解释边界','', '；'.join(decisions)+'。不按这些测试表现重选C、种子、描述或增强力度。','',
'AUROC、pAUC、检出率、误报分别判断。图片整体特征可能只记视觉类别；图片转文字同时引入OCR/生成先验与更长文本，要结合等长中性描述和去OCR/去描述对照；有图无图差不是严格因果效应；ROI注意力不是危险性证明；扰动不稳定性也会出现在正常OCR/低照图片。类目分组详见[subgroups.json](subgroups.json)，小组仅描述，不再搜索。','',
'原图尺寸未显式提供给检测头，但anyres预处理仍可能暴露几何/来源捷径。按原场景去重的配对同尺寸审计见[image-geometry-audit.json](image-geometry-audit.json)。本轮未用额外配对尺寸归一化实验排除这些捷径，因此不能把增益全部归因于语义理解。','',
'阈值只由82条正常cal的0基77号顺序统计量固定，严格score>阈值。校准样本少且有关联，误报率粒度较粗，不用test再校准。pAUC为sklearn标准化FPR≤10%部分面积。','',
'## 成本与可靠性','',
f"冻结联合提取的实际操作计数：`{json.dumps(ex['forward_counts'],ensure_ascii=False)}`；记录操作总秒数：`{json.dumps(ex['seconds_by_operation'],ensure_ascii=False)}`。记录操作计数完整性：{cost_complete}；峰值已分配显存{ex['maximum_allocated_gpu_bytes']/1024**3:.2f}GiB。这是联合提取，不能当成某个候选的独立部署延迟；模型加载、部分CPU/I/O和OCR并非全部包含在该秒数中。",'',
f"OCR失败{ex['ocr_failures']}条，空OCR文字{ex['empty_ocr']}条；失败未按标签删除。描述由冻结LLaVA贪心生成，最多40新token，不生成用户任务答案，原文不公开。详见[cost-audit.json](cost-audit.json)。",'']
 if (OUT/'benchmark.json').exists():
  b=json.loads((OUT/'benchmark.json').read_text());assert b['passed'] and b['evaluation_sha256']==proof['evaluation_sha256'];lines+=['固定8条训练样本的独立部署示例测时：模型已加载、文件缓存热、清空文字/OCR特征缓存，包含特征与首种子头；不是总体延迟估计。','', '|方案|中位秒数|示例p95秒数|最大已分配GiB|','|---|---|---|---|']
  for name in [BASE]+ARMS:
   rr=[x for x in b['records'] if x['name']==name];ss=[x['seconds'] for x in rr];lines.append(f"|{name}|{np.median(ss):.3f}|{np.quantile(ss,.95):.3f}|{max(x['maximum_allocated_gpu_bytes'] for x in rr)/1024**3:.2f}|")
  lines+=['','完整逐样本测时与预测复现差见[benchmark.json](benchmark.json)。','']
 lines+=['独立只读审查指出断点恢复与成本累计隐患；保留原冻结训练代码，追加恢复入口：有封印则重用原头，不重新训练或选择；历史成本缺失必须标为下界。当前是否实际恢复见成本审计。','',
f"独立核验重载126原头并绑定42最终条目，重新计算训练侧PCA/质心/标准化、全部分数/阈值/四指标、10个组区间。预测重放最大差{proof['prediction_max_difference']:.3g}；另重新加载视觉模型，在训练和两批留出的6条冻结输入重做全部9个F分支、OCR及描述，记录见[verification.json](verification.json)。",'',
'[执行协议](../../docs/ei2-images-protocol-2026-11-05.md) · [初始梳理](../../docs/ei2-image-information-design-2026-11-05.md) · [源码锁](source-lock.json) · [清单](manifest.json) · [评分前封印](pre-evaluation-seal.json) · [提取审计](extraction.json)','',
f"evaluation SHA256：`{proof['evaluation_sha256']}`；scores SHA256：`{proof['scores_sha256']}`。公开内容不含图片、OCR/描述原文、视觉/隐藏向量、模型/检测头或凭据。"]
 lines+=['',f"全部{coverage['rows_checked']}条输入覆盖审计通过：原图文最长{coverage['main_expanded_max']} token，描述生成上下文加40个新token最长{coverage['caption_context_plus_40_new_tokens_max']}，均未触及4096上限；原输入token哈希逐条重建一致。详见[input-coverage-audit.json](input-coverage-audit.json)。"]
 (OUT/'report-2026-11-05.md').write_bytes(('\n'.join(lines)+'\n').encode());print('VERIFIED_IMAGE_REPORT_AND_DESCRIPTIVE_SUBGROUPS_RENDERED')
if __name__=='__main__':main()
