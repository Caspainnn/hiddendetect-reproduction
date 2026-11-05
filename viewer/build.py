"""Build the same experiment tree offline, without a personal reading collection."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parent
sys.path.insert(0,str(HERE/'templates'))
import experiment_mindmap as model
import build_experiments as builder
model.ROOT=HERE;model.REPO=REPO;builder.ROOT=HERE
builder.build({'papers':[{'pid':'hiddendetect','title':'HiddenDetect'}]})
page=HERE/'experiments/detail.html';text=page.read_text(encoding='utf-8')
marker='\nrender();\n</script>'
assert marker in text
text=text.replace(marker,"\nfor(const row of experiments){if(row.source.startsWith('../../docs/'))row.source='docs/project/'+row.source.slice('../../docs/'.length);}\nrender();\n</script>",1)
text=text.replace("paperEntry.href='../index.html#paper='+linkedPaper.pid","paperEntry.href='https://aclanthology.org/2025.acl-long.724/'")
text=text.replace("selected.paper+' · 论文精读 →'","selected.paper+' · 论文入口 →'")
text=text.replace('../readings/论文阅读/网页/favicon.ico','../favicon.ico')
page.write_bytes(text.encode())
home=HERE/'experiments/index.html';home.write_bytes(home.read_text(encoding='utf-8').replace('../readings/论文阅读/网页/favicon.ico','../favicon.ico').encode())
(HERE/'index.html').write_bytes(b'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=experiments/detail.html?id=hiddendetect"><title>HiddenDetect experiments</title><a href="experiments/detail.html?id=hiddendetect">Open experiments</a>')
(REPO/'index.html').write_bytes(b'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=viewer/"><title>HiddenDetect experiments</title><a href="viewer/">Open experiment viewer</a>')
print('EXPERIMENT_VIEWER_BUILT',page)
