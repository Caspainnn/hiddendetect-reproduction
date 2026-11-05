const ExperimentProjects=(()=>{
const key='paper-experiment-prototype-v1',initial={id:'hiddendetect',paper:'HiddenDetect',title:'HiddenDetect论文优化'};
function all(){try{const rows=JSON.parse(localStorage.getItem(key));if(Array.isArray(rows)){const clean=rows.filter(r=>r&&typeof r.id==='string'&&/^[a-z0-9-]+$/.test(r.id)&&typeof r.paper==='string'&&typeof r.title==='string').map(r=>({id:r.id,paper:r.paper.slice(0,100),title:r.title.slice(0,100)}));if(clean.length)return clean;}}catch{}return [{...initial}];}
function save(rows){try{localStorage.setItem(key,JSON.stringify(rows));return true;}catch{return false;}}
function selected(){const q=new URLSearchParams(location.search),id=q.get('id')||initial.id,row=all().find(r=>r.id===id);return {id,paper:(q.get('paper')||row?.paper||'未命名论文').slice(0,100),title:(q.get('title')||row?.title||'未命名论文优化').slice(0,100)};}
function url(row){return 'detail.html?'+new URLSearchParams({...row,theme:document.body.classList.contains('dark')?'dark':'light'}).toString();}
return {all,save,selected,url};})();
