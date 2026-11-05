"""Build the approved experiment UI from maintained templates, offline."""
import json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

from experiment_mindmap import build_oc,build_ei2,build_ei1

def seed():
    prototype=ROOT/'design/2026-11-02-experiments'
    for source,target in [('index.html','experiment-home.html'),('detail.html','experiment-detail.html'),('projects.js','experiment-projects.js')]:
        path=HERE/target
        assert not path.exists(),path
        path.write_text((prototype/source).read_text(encoding='utf-8'),encoding='utf-8')

def build(data):
    out=ROOT/'experiments';out.mkdir(exist_ok=True)
    store=(HERE/'experiment-projects.js').read_text(encoding='utf-8').replace('paper-experiment-prototype-v1','paper-experiments-v1')
    library=json.dumps([{'pid':p['pid'],'title':p['title']} for p in data['papers']],ensure_ascii=False).replace('<','\\u003c')
    for template,name in [('experiment-home.html','index.html'),('experiment-detail.html','detail.html')]:
        html=(HERE/template).read_text(encoding='utf-8')
        html=html.replace('<script src="projects.js"></script>','<script>'+store+'</script>')
        html=html.replace('../../index.html','../index.html').replace('../../experiments/HiddenDetect/','HiddenDetect/')
        html=html.replace('HTML 原型 · 2026.11.02','论文实验工作台')
        html=html.replace('</head>','<link rel="icon" href="../readings/论文阅读/网页/favicon.ico"></head>')
        html=html.replace('</head>', '''<style>body.dark{color-scheme:dark;--paper:#252925;--bg:#1e211e;--text:#e5e7dc;--muted:#b0b9ac;--green:#b2c7a1;--soft:#303b2d;--line:#41493e;background:var(--paper)}body.dark .card,body.dark input,body.dark .cancel,body.dark .project{background:var(--paper)}body.dark .cover{border-color:var(--line)}</style></head>''')
        html=html.replace('</body>', '''<script>if(new URLSearchParams(location.search).get('theme')==='dark')document.body.classList.add('dark');</script></body>''')
        html=html.replace('href="index.html" aria-current="page"','href="../index.html#experiments" aria-current="page"')
        if name=='detail.html':
            html=html.replace('<!-- EXPERIMENT_MINDMAP_VIEW -->',(HERE/'experiment-mindmap.html').read_text(encoding='utf-8'),1)
            directions,oc_rows=build_oc()
            ei2_questions,ei2_rows=build_ei2()
            ei1_questions,ei1_rows=build_ei1()
            html=html.replace('</head>','<style>'+(HERE/'experiment-mindmap.css').read_text(encoding='utf-8')+'</style></head>',1)
            numbering=json.loads((ROOT.parent/'docs/experiment-numbering.json').read_text(encoding='utf-8'))
            script='const experimentNumbering='+json.dumps(numbering,ensure_ascii=False).replace('<','\\u003c')+';\n'
            script+='const ocMindDirections='+json.dumps(directions,ensure_ascii=False).replace('<','\\u003c')+';\n'
            script+="experiments.splice(experiments.findIndex(x=>x.group==='OC'),1,..."+json.dumps(oc_rows,ensure_ascii=False).replace('<','\\u003c')+");\n"
            script+='const ei2MindQuestions='+json.dumps(ei2_questions,ensure_ascii=False).replace('<','\\u003c')+';\n'
            script+='experiments.push(...'+json.dumps(ei2_rows,ensure_ascii=False).replace('<','\\u003c')+');\n'
            script+='const ei1MindQuestions='+json.dumps(ei1_questions,ensure_ascii=False).replace('<','\\u003c')+';\n'
            script+="experiments.splice(experiments.findIndex(x=>x.topic==='centroid'),1,..."+json.dumps(ei1_rows,ensure_ascii=False).replace('<','\\u003c')+");\n"
            script+=(HERE/'experiment-mindmap.js').read_text(encoding='utf-8').replace('/* EI_TAXONOMY_V2 */',(HERE/'experiment-ei-taxonomy.js').read_text(encoding='utf-8'))
            html=html.replace('/* EXPERIMENT_MINDMAP */',script,1)
            html=html.replace('href="index.html" style="display:inline-block','href="../index.html#experiments" style="display:inline-block')
            start=html.index("if(selected.id==='hiddendetect'")
            end=html.index('\ndocument.title=',start)
            html=html[:start]+'''const library=PAPER_LIBRARY;
const linkedPaper=library.find(p=>p.title.trim().toLowerCase()===selected.paper.trim().toLowerCase());
if(linkedPaper){paperEntry.textContent=selected.paper+' · 论文精读 →';paperEntry.href='../index.html#paper='+linkedPaper.pid;paperEntry.setAttribute('aria-label','打开 '+selected.paper+' 论文精读');}
else{const label=document.createElement('span');label.className='project';label.textContent=selected.paper;paperEntry.replaceWith(label);}
'''.replace('PAPER_LIBRARY',library)+html[end:]
        else:
            # Isolate card styles from the reading workspace; card links enter full pages.
            html=html.replace('</head>','<style>body.embedded header{display:none}body.embedded main{padding:18px 0 30px;max-width:none}</style></head>')
            html=html.replace('</body>','''<script>
if(new URLSearchParams(location.search).get('embed')==='1'){
 document.body.classList.add('embedded');
 window.addEventListener('message',e=>{if(e.source===parent&&e.data?.type==='experiment-theme'){document.body.classList.toggle('dark',e.data.dark===true);render();}});
 const base=document.createElement('base');base.target='_top';document.head.append(base);
 const notify=()=>parent.postMessage({type:'experiment-height',height:document.documentElement.scrollHeight},location.protocol==='file:'?'*':location.origin);
 new ResizeObserver(notify).observe(document.body);notify();
}
</script></body>''')
        (out/name).write_text(html,encoding='utf-8')

if __name__=='__main__':
    import sys
    if '--seed' in sys.argv:seed()
