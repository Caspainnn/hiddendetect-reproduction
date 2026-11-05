"""Freeze image/intent connected groups and all roles before model extraction."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedShuffleSplit
ROOT=Path(__file__).resolve().parents[2];ACQ=ROOT/'cache/ei2-image-acquisition';RAW=ACQ/'raw';OUT=ROOT/'results/ei2-images';CACHE=ROOT/'cache/ei2-images'
def sha(data):return hashlib.sha256(data).hexdigest()
def textkey(s):return re.sub(r'\s+',' ',s.strip().lower())
def phash(im):
 a=np.array(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),float);v=dctn(a,type=2,norm='ortho')[:8,:8].ravel();return sum(int(x>np.median(v[1:]))<<j for j,x in enumerate(v))
def main():
 OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists(),'Inspect frozen manifest before preparing again'
 acquisition=json.loads((ACQ/'acquisition.json').read_text());data=json.loads((RAW/'combined.json').read_text());imageinfo={}
 for record in acquisition['files']:
  if record['path']=='combined.json':continue
  p=RAW/record['path'];assert sha(p.read_bytes())==record['sha256'];im=Image.open(p).convert('RGB');imageinfo[record['path']]={'sha256':record['sha256'],'width':im.width,'height':im.height,'pixel_sha256':sha(im.tobytes()),'phash64':phash(im)}
 scenes=[]
 for i,q in enumerate(data['chat']):scenes.append({'key':f'chat-{i:03d}','domain':'chat','category':q['Type'],'intent':textkey(q['intent']),'images':['chat/'+q['safe_image_path'],'chat/'+q['unsafe_image_path']],'questions':[q['queries'],q['queries']]})
 for i,q in enumerate(data['embodied']):scenes.append({'key':f'embodied-{i:03d}','domain':'embodied','category':q['category'],'intent':textkey(q['task']),'images':['embodied/'+q['safe'],'embodied/'+q['unsafe']],'questions':[[q['safe_instruction']],[q['unsafe_instruction']]]})
 parent=list(range(len(scenes)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(i,j):
  a,b=find(i),find(j)
  if a!=b:parent[max(a,b)]=min(a,b)
 imageowners=defaultdict(list);intentowners={};edges=[]
 for i,s in enumerate(scenes):
  for name in s['images']:imageowners[name].append(i)
  if s['intent'] in intentowners:union(i,intentowners[s['intent']]);edges.append({'kind':'same_intent','a':scenes[intentowners[s['intent']]]['key'],'b':s['key']})
  else:intentowners[s['intent']]=i
 names=sorted(imageinfo)
 for a,name in enumerate(names):
  for other in names[:a]:
   x,z=imageinfo[name],imageinfo[other]
   if x['sha256']==z['sha256'] or x['pixel_sha256']==z['pixel_sha256'] or (x['phash64']^z['phash64']).bit_count()<=4:
    for i in imageowners[name]:
     for j in imageowners[other]:union(i,j)
    edges.append({'kind':'same_or_near_image','a':name,'b':other})
 clusters=defaultdict(list)
 for i,s in enumerate(scenes):clusters[find(i)].append(s)
 groupid={i:'g-'+sha('|'.join(sorted(s['key'] for s in group)).encode())[:16] for i,group in clusters.items()};roles={};overlap=[];eligible=[]
 for i,group in clusters.items():
  if any(s['domain']=='embodied' for s in group):
   roles[groupid[i]]='task_test';overlap.extend(s['key'] for s in group if s['domain']=='chat')
  else:eligible.append((groupid[i],group[0]['category']))
 eligible.sort();indices=np.arange(len(eligible));categories=np.array([x[1] for x in eligible]);remaining,test=next(StratifiedShuffleSplit(1,test_size=.2,random_state=20261105).split(indices,categories));rest,cal=next(StratifiedShuffleSplit(1,test_size=1/6,random_state=20261106).split(remaining,categories[remaining]));cal=remaining[cal];remaining=remaining[rest];train,dev=next(StratifiedShuffleSplit(1,test_size=.2,random_state=20261107).split(remaining,categories[remaining]));train,dev=remaining[train],remaining[dev]
 for role,ix in [('train',train),('dev',dev),('cal',cal),('test',test)]:
  for j in ix:roles[eligible[int(j)][0]]=role
 rows=[];private=[]
 for i,s in enumerate(scenes):
  group=groupid[find(i)];role=roles[group]
  if s['domain']=='chat' and role=='task_test':continue
  for label in [0,1]:
   if role=='cal' and label==1:continue
   name=s['images'][label];info=imageinfo[name]
   for qi,prompt in enumerate(s['questions'][label]):
    key=s['key']+f'-q{qi}-y{label}';public={'id':key,'group_id':group,'scene_id':s['key'],'role':role,'label':label,'domain':s['domain'],'category':s['category'],'image_sha256':info['sha256'],'image_pixel_sha256':info['pixel_sha256'],'image_phash64':info['phash64'],'width':info['width'],'height':info['height'],'prompt_sha256':sha(prompt.encode())};rows.append(public);private.append(public|{'prompt':prompt,'image_path':name})
 order=sorted(range(len(rows)),key=lambda i:(['train','dev','cal','test','task_test'].index(rows[i]['role']),rows[i]['id']));rows=[rows[i] for i in order];private=[private[i] for i in order]
 assert all(len({r['role'] for r in rows if r['group_id']==g})==1 for g in roles);assert len({r['id'] for r in rows})==len(rows);assert len([r for r in rows if r['role']=='cal'])>=40
 manifest={'run_id':'EI-IM1','dataset':'MSSBench','revision':acquisition['revision'],'acquisition_sha256':sha((ACQ/'acquisition.json').read_bytes()),'prepare_code_sha256':sha(Path(__file__).read_bytes()),'samples':rows,'counts':dict(Counter(r['role']+':'+str(r['label']) for r in rows)),'groups':dict(Counter(roles.values())),'excluded_overlap_chat_scenes':overlap,'label_definition':'Official situational context requires safety intervention (1) versus safe context (0); not attack success or image-content-only moderation','official_split':'No official train/test in release; scene/image/intent connected components split prospectively','transferred_task':'Embodied is same-collection held-out task, not an external source','seen_evaluation':False}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(OUT/'grouping.json').write_text(json.dumps({'edges':edges,'scene_to_group':{s['key']:groupid[find(i)] for i,s in enumerate(scenes)},'same_intent_and_image_components':True,'near_image_hamming_threshold':4,'generic_question_not_grouped':'Identical generic questions occur with both labels; original scene/intent and image determine grouping'},indent=2)+'\n');(CACHE/'records.json').write_text(json.dumps(private,indent=2)+'\n');print('FROZEN_IMAGE_ROLES',manifest['counts'],'GROUPS',manifest['groups'],'OVERLAP_EXCLUDED',len(overlap),flush=True)
if __name__=='__main__':main()
