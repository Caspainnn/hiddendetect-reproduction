// A single nested tree: a button folds its own descendants without replacing siblings.
const plainText = value => value.replace(/<br\s*\/?\s*>/gi,'\n').replace(/<[^>]*>/g,'');
function metricParts(text){
 const parts=[];let end=0;
 for(const match of text.matchAll(/(\d+(?:\.\d+)?%?)\s*→\s*(\d+(?:\.\d+)?%?)/g)){
  const candidateStart=match.index+match[0].lastIndexOf(match[2]);
  parts.push({text:text.slice(end,candidateStart)});
  const lowerBetter=/(?:误报率|误报|FPR)\s*$/i.test(text.slice(0,match.index));
  const before=parseFloat(match[1]),after=parseFloat(match[2]);
  parts.push({text:match[2],improved:lowerBetter?after<before:after>before});
  end=match.index+match[0].length;
 }
 parts.push({text:text.slice(end)});return parts;
}
function appendMetricText(element,text){
 for(const part of metricParts(text)){
  const span=document.createElement(part.improved?'strong':'span');
  if(part.improved)span.className='metric-improvement';
  span.textContent=part.text;element.append(span);
 }
}
const eiDirections = [
 ['centroid','EI-1：质心距离：多一条距离线索会更准吗？'],
 ['logits','EI-2：完整logits：保留更多输出信息是否有用？'],
 ['position','EI-3：读取位置：危险线索藏在哪里？'],
 ['layers','EI-4：选层：哪些层更有用？']
].map(([topic,label]) => ({id:topic,label,kind:'direction',children:experiments.filter(x=>x.group==='EI'&&x.topic===topic).map((x,i)=>({id:topic+'-q'+i,label:x.why,kind:'question',children:[{label:x.name,kind:'experiment',children:[{label:'方法：'+x.how,kind:'method'},{label:'结果：'+plainText(x.metrics),kind:'result'},{label:'结论：'+x.verdict+'\n'+x.batch,kind:'evidence'}]}]}))}));
const centroidDirection=eiDirections.find(x=>x.id==='centroid');
centroidDirection.label='隐藏状态：除了F，还能提供什么线索？';
centroidDirection.children=ei1MindQuestions;
const vocabularyDirection=eiDirections.find(x=>x.id==='logits');
vocabularyDirection.label='EI-2：词表信息保留\n旧OC-4-3／迁入EI';
vocabularyDirection.children=[
 ...vocabularyDirection.children.filter(x=>!ei2MindQuestions.some(q=>x.children[0].label.startsWith(q.children[0].label.split('：')[0]+'：'))),
 ...ei2MindQuestions
];
/* EI_TAXONOMY_V2 */
const mindRoot={id:'root',label:'怎样把危险输入\n识别得更准？',kind:'root',children:[
 {id:'ei',label:'EI：调整提供给检测器的信息',kind:'group',children:eiV2Directions},
 readoutDirection,
 {id:'oc',label:'OC：信息相同，改变加工与判分',kind:'group',tone:'oc',children:ocMindDirections.map(x=>({...x,kind:'direction'}))}
]};
const defaultOpen=new Set(['root','oc','oc3','oc-3-1-question','oc-3-1']);
const mindViewport=document.getElementById('mind-viewport');
const mindCanvas=document.getElementById('mind-canvas');
let mindCounter=0,mindZoom=80;
function makeMindNode(data,parentId){
 const li=document.createElement('li');li.className='mind-item'+(data.tone==='oc'?' mind-oc':'');li.dataset.kind=data.kind||'direction';
 const id=data.id||'mind-'+(++mindCounter);li.dataset.node=id;
 const children=data.children||[],branch=children.length>0;
 const box=document.createElement(branch?'button':'div');box.className='mind-node';
 const label=document.createElement('span');
 if(data.kind==='result')appendMetricText(label,data.label);else label.textContent=data.label;
 box.append(label);
 if(branch){
  box.type='button';box.dataset.node=id;if(parentId)box.dataset.parent=parentId;
  box.setAttribute('aria-expanded',String(defaultOpen.has(id)));box.setAttribute('aria-controls','children-'+id);
  const toggle=document.createElement('span');toggle.className='mind-toggle';toggle.setAttribute('aria-hidden','true');box.append(toggle);
  const ul=document.createElement('ul');ul.className='mind-children';ul.id='children-'+id;ul.hidden=!defaultOpen.has(id);
  for(const child of children)ul.append(makeMindNode(child,id));
  toggle.textContent=defaultOpen.has(id)?'−':String(children.length);li.append(box,ul);
 }else li.append(box);
 return li;
}
function updateMindToggle(button){
 const opened=button.getAttribute('aria-expanded')==='true';
 const children=document.getElementById(button.getAttribute('aria-controls'));
 button.querySelector('.mind-toggle').textContent=opened?'−':String(children.children.length);
}
function setMindOpen(button,open){
 button.setAttribute('aria-expanded',String(open));
 document.getElementById(button.getAttribute('aria-controls')).hidden=!open;updateMindToggle(button);
}
function resetMind(mode){
 mindCanvas.querySelectorAll('button.mind-node').forEach(button=>{
  const id=button.dataset.node;
  setMindOpen(button,mode==='all'||(mode==='directions'?['root','ei','oc'].includes(id):defaultOpen.has(id)));
 });
 mindViewport.scrollTop=0;mindViewport.scrollLeft=0;
}
const mindList=document.createElement('ul');mindList.append(makeMindNode(mindRoot));mindCanvas.append(mindList);
mindCanvas.addEventListener('click',event=>{
 const button=event.target.closest('button.mind-node');if(!button)return;
 const before=button.getBoundingClientRect();
 setMindOpen(button,button.getAttribute('aria-expanded')!=='true');
 const after=button.getBoundingClientRect();
 mindViewport.scrollTop+=after.top-before.top;mindViewport.scrollLeft+=after.left-before.left;
});
mindCanvas.addEventListener('keydown',event=>{
 const button=event.target.closest('button.mind-node');if(!button)return;
 if(event.key==='ArrowLeft'){
  event.preventDefault();
  if(button.getAttribute('aria-expanded')==='true')setMindOpen(button,false);
  else if(button.dataset.parent)mindCanvas.querySelector('button[data-node="'+button.dataset.parent+'"]').focus();
 }else if(event.key==='ArrowRight'){
  event.preventDefault();
  if(button.getAttribute('aria-expanded')==='false')setMindOpen(button,true);
  else document.getElementById(button.getAttribute('aria-controls')).querySelector('button.mind-node')?.focus();
 }
});
document.getElementById('mind-expand-all').onclick=()=>resetMind('all');
document.getElementById('mind-collapse').onclick=()=>resetMind('directions');
document.getElementById('mind-reset').onclick=()=>{resetMind('default');setMindZoom(80)};
function setMindZoom(value,anchor){
 const previous=mindZoom;
 mindZoom=Math.min(200,Math.max(40,Math.round(value)));
 const ratio=mindZoom/previous;
 const left=anchor?(mindViewport.scrollLeft+anchor.x)*ratio-anchor.x:mindViewport.scrollLeft;
 const top=anchor?(mindViewport.scrollTop+anchor.y)*ratio-anchor.y:mindViewport.scrollTop;
 mindCanvas.style.zoom=mindZoom+'%';document.getElementById('mind-zoom-value').textContent=mindZoom+'%';
 if(anchor){mindViewport.scrollLeft=left;mindViewport.scrollTop=top}
}
document.getElementById('mind-zoom-out').onclick=()=>setMindZoom(mindZoom-10);
document.getElementById('mind-zoom-in').onclick=()=>setMindZoom(mindZoom+10);
const mindSection=document.getElementById('mind-section');
const mindFullscreenButton=document.getElementById('mind-fullscreen');
let mindFullscreen=false,mindNativeFullscreen=false,mindSavedOverflow='';
function setMindFullscreen(active){
 if(active===mindFullscreen)return;
 if(active)mindSavedOverflow=document.body.style.overflow;
 document.body.style.overflow=active?'hidden':mindSavedOverflow;
 mindFullscreen=active;mindSection.classList.toggle('mind-fullscreen',active);
 mindFullscreenButton.textContent=active?'退出全屏（Esc）':'全屏查看';
 mindFullscreenButton.setAttribute('aria-pressed',String(active));
 if(!active){finishMindDrag();mindFullscreenButton.focus()}
}
async function exitMindFullscreen(){
 if(document.fullscreenElement===mindSection){try{await document.exitFullscreen()}catch{}}
 setMindFullscreen(false);
}
mindFullscreenButton.onclick=async()=>{
 if(mindFullscreen){await exitMindFullscreen();return;}
 setMindFullscreen(true);
 // Keep the window-filling view if browser fullscreen is unavailable for a local page.
 if(mindSection.requestFullscreen){try{await mindSection.requestFullscreen()}catch{}}
};
document.addEventListener('fullscreenchange',()=>{
 if(document.fullscreenElement===mindSection){mindNativeFullscreen=true;setMindFullscreen(true)}
 else if(mindNativeFullscreen){mindNativeFullscreen=false;setMindFullscreen(false)}
});
document.addEventListener('keydown',event=>{
 if(event.key==='Escape'&&mindFullscreen){event.preventDefault();exitMindFullscreen()}
});
let mindDrag=null,mindSuppressClick=false;
mindViewport.addEventListener('pointerdown',event=>{
 if(!mindFullscreen||event.button!==0||event.isPrimary===false)return;
 mindSuppressClick=false;
 mindDrag={id:event.pointerId,x:event.clientX,y:event.clientY,left:mindViewport.scrollLeft,top:mindViewport.scrollTop,moved:false};
});
mindViewport.addEventListener('pointermove',event=>{
 if(!mindDrag||event.pointerId!==mindDrag.id)return;
 const dx=event.clientX-mindDrag.x,dy=event.clientY-mindDrag.y;
 if(!mindDrag.moved&&Math.hypot(dx,dy)<5)return;
 if(!mindDrag.moved){
  mindDrag.moved=true;mindSuppressClick=true;
  mindViewport.setPointerCapture(event.pointerId);mindViewport.classList.add('mind-dragging');
 }
 event.preventDefault();
 mindViewport.scrollLeft=mindDrag.left-dx;mindViewport.scrollTop=mindDrag.top-dy;
});
function finishMindDrag(event){
 if(!mindDrag||(event&&event.pointerId!==mindDrag.id))return;
 const id=mindDrag.id;mindDrag=null;mindViewport.classList.remove('mind-dragging');
 if(mindViewport.hasPointerCapture(id))mindViewport.releasePointerCapture(id);
}
mindViewport.addEventListener('pointerup',finishMindDrag);
mindViewport.addEventListener('pointercancel',finishMindDrag);
mindViewport.addEventListener('lostpointercapture',finishMindDrag);
// A drag that starts on a node must not also fold it when the mouse is released.
mindViewport.addEventListener('click',event=>{
 if(mindSuppressClick&&event.detail!==0){mindSuppressClick=false;event.preventDefault();event.stopPropagation()}
},true);
mindViewport.addEventListener('wheel',event=>{
 if(!mindFullscreen||event.deltaY===0)return;
 event.preventDefault();finishMindDrag();
 const rect=mindViewport.getBoundingClientRect();
 const pixels=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?mindViewport.clientHeight:1);
 const step=Math.max(-100,Math.min(100,pixels));
 setMindZoom(mindZoom*Math.exp(-step*.0015),{x:event.clientX-rect.left,y:event.clientY-rect.top});
},{passive:false});
