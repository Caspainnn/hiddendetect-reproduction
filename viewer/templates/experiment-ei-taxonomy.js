// Current EI interface taxonomy. Archive IDs and frozen experiment records stay intact.
const eiLeaf=(label,kind='evidence')=>({label,kind,children:[]});
const archiveQuestion=(original,label,trace)=>{
 const q=structuredClone(original);q.children[0].label=label;
 q.children[0].children.push(eiLeaf('编号追溯：'+trace+'；本次只调整展示归属，不是新增运行。'));
 return q;
};
const centroidIds=experimentNumbering.centroid;
const cOnly=archiveQuestion(ei1MindQuestions[0],centroidIds.replace_history+'：只看质心C（历史C32）','旧EI-1历史C32／signal-selection → '+centroidIds.replace_history+'；仅登记编号，冻结键C32不改');
const fPlusC=archiveQuestion(ei1MindQuestions[1],centroidIds.add_history+'：原F＋质心C（历史all64）','旧EI-1历史all64／signal-selection → '+centroidIds.add_history+'；仅登记编号，冻结键all64不改');
fPlusC.children[0].children.push(eiLeaf('问题承接：先看C能否判断，再验证加回原F是否更好。历史all64相对C32的额外优势未确认；不能用相对F32的区间代替。'));
const historyLogits=vocabularyDirection.children.find(q=>q.children[0].label.startsWith('EI-2 ')||q.children[0].label.startsWith('EI-2：读取'));
if(!historyLogits)throw new Error('Historical complete-logits record missing');
const logitsOnly=archiveQuestion(historyLogits,centroidIds.logits_history+'：只看完整logits（历史first_logit）','旧EI-2历史first_logit → '+centroidIds.logits_history+'改变信息；冻结键first_logit不改');
logitsOnly.children[0].children.unshift(eiLeaf('输入核对：最终层输入末位的32000维完整logits＋逻辑回归；原对照是32层F32监督探针，不是14层完整词表。','method'));
const pendingQuestion=(id,question,title,method,boundary)=>({id,label:question,kind:'question',children:[{id:id+'-experiment',label:title,kind:'experiment',children:[eiLeaf('方法：'+method,'method'),eiLeaf('结果：待验证，暂无该新方案的已完成成绩。','result'),eiLeaf(boundary)]}]});
const centroidPending=[
 {id:centroidIds.add_matched,question:'固定同样14层，原F加上对应质心C，仍能提高判断吗？',title:'原14F＋质心C14',method:'沿用原C定义，只在训练侧拟合与原F对应层的安全/危险质心；原14F＋C14，重训同一L2头。',boundary:'直接对照是同批原14F＋同头；原层、末位、标签和调参预算固定。对应隐藏状态与F层的对齐须执行前核验；历史all64不是本项成绩。'},
 {id:centroidIds.replace_matched,question:'只看同样14层的质心C，不提供F，判断效果如何？',title:'只看质心C14',method:'保留与原14F对应的14层和末位，训练侧拟合质心C14，检测头只接收C14。',boundary:'直接对照是同批原14F＋同一L2头；与F＋C14的增量另作配对比较。不借用历史32层C32结果。'}
];
function centroidGroup(id,question,history,pending){
 const verified=ei1MindQuestions.find(q=>q.id===pending.id.toLowerCase()+'-question');
 const next=verified?structuredClone(verified):pendingQuestion(pending.id.toLowerCase()+'-question',pending.question,pending.id+'：'+pending.title+'（待验证）',pending.method,pending.boundary);next.children[0].id=pending.id.toLowerCase();
 return {id:id.toLowerCase()+'-question',label:question,kind:'question',children:[{id:id.toLowerCase(),label:id+'：'+(id===centroidIds.add_group?'原F＋质心线索':'只看质心线索'),kind:'direction',children:[history,next]}]};
}
const eiV2Directions=[
 {id:'ei-v2-1',label:'EI-1：改变F\n换一种构造或压缩方式',kind:'direction',children:[
  pendingQuestion('ei-v2-1-compression','原来压成F时丢掉的线索，换一种压缩能保留吗？','EI-1：重构F′（待验证）','从原词表信息构造替代F′，不把已压缩的F展开冒充新增信息；固定位置、层、头和选择预算。','关联已实现的旧OC-2单层评分实验：均值、平滑最大、概率质量。它们保留原OC-2运行和结果；此处仅交叉引用，不重复立项或挪用成绩。')
 ]},
 {id:'ei-v2-2',label:'EI-2：添加信息\n原F＋补充线索',kind:'direction',children:[...structuredClone(ei2MindQuestions),centroidGroup(centroidIds.add_group,'词表分数之外，质心还能补上什么？',fPlusC,centroidPending[0])]},
 {id:'ei-v2-3',label:'EI-3：改变信息\n完全不提供原F',kind:'direction',children:[centroidGroup(centroidIds.replace_group,'离安全或危险样本中心有多近，能替代F吗？',cOnly,centroidPending[1]),logitsOnly]},
 {id:'ei-v2-4',label:'EI-4：以上混合\n组合已验证的改动',kind:'direction',children:[
  pendingQuestion('ei-v2-4-combination','分别有效的改动组合后，是否还有额外帮助？','EI-4：混合验证（待验证）','只在训练/选择侧挑组件，冻结后逐项消融并用未见数据确认；每个组合写清哪些改动叠加。','尚无按这套新四分类完成的混合验证。旧EI-4-1/2是选层实验，不是新EI-4的成绩；简单原F＋C首先归EI-2，不重复算混合成功。')
 ]}
];
function numberReadout(direction){
 const copy=structuredClone(direction);
 copy.label=experimentNumbering.readout_groups[copy.id];
 for(const question of copy.children){
  const exp=question.children[0],record=experimentNumbering.readout.find(x=>x.legacy_label===exp.label);
  if(!record)throw new Error('Unnumbered readout experiment: '+exp.label);
  exp.id=record.id.toLowerCase();exp.label=record.id+'：'+record.title;
  exp.children.push(eiLeaf('编号追溯：'+record.legacy_ids+'／'+record.run+' → '+record.id+'；只改登记与展示，原冻结键、批次和指标不改。'));
 }
 return copy;
}
const readoutDirection={id:'common-readout',label:'CFG：共同读取设置\n在哪里读、读哪些层？',kind:'group',children:eiDirections.filter(d=>['position','layers'].includes(d.id)).map(numberReadout)};
for(const row of experiments){
 if(row.group!=='EI')continue;
 row.archivedName=row.name;
 if(row.topic==='centroid')row.name=row.name.includes('C32')?centroidIds.replace_history+'：只看质心C（历史C32）':centroidIds.add_history+'：原F＋质心C（历史all64）';
 else if(row.topic==='logits'&&!/^EI-2-[1-6]/.test(row.name))row.name=centroidIds.logits_history+'：只看完整logits（历史first_logit）';
 else if(['position','layers'].includes(row.topic)){
  const record=experimentNumbering.readout.find(x=>x.legacy_label===row.name);if(!record)throw new Error('Missing readout registry: '+row.name);
  row.group='CFG';row.name=record.id+'：'+record.title;row.batch+=' 编号追溯：'+record.legacy_ids+'／'+record.run+'；冻结记录不改。';
 }
 if(row.name!==row.archivedName)row.batch+=' 新四分类展示；旧名称：'+row.archivedName+'。';
}
for(const p of centroidPending)if(!experiments.some(row=>row.name.startsWith(p.id+'：')))experiments.push({topic:'centroid',group:'EI',name:p.id+'：'+p.title,why:p.question,how:p.method,metrics:'待验证；AUROC、低误报pAUC、检出率、误报暂无新成绩。',batch:'固定14层的新方案，未启动；历史32层结果不搬用。',verdict:p.boundary,status:'待验证',source:'docs/experiment-numbering.md'});
for(const [topic,name,why] of [
 ['ei-v2-1','EI-1：重新构造F′（待验证）','换一种压缩或构造方式，能否保留原F丢掉的线索？'],
 ['ei-v2-4','EI-4：混合验证（待验证）','分别有效的改动组合后，是否还有额外帮助？']
])experiments.push({topic,group:'EI',name,why,how:'评分前明确输入、头、训练和调参预算。已有相关实验仅作关联；具体方案另行冻结。',metrics:'待验证，暂无本新方案的已完成成绩。',batch:'新四分类；旧选层成绩不能当作混合成绩。',verdict:'只登记方向与旧结果归属，本次不执行新实验。',status:'待验证',source:'../../docs/ei-interface-taxonomy.md'});
