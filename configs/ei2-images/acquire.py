"""Public official MSSBench acquisition, pinned and content-verified."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time,threading
import requests
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'cache/ei2-image-acquisition';REV='12396d24157427b671d97f23a2e370fca98fdf55'
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
 CACHE.mkdir(parents=True,exist_ok=True);base='https://huggingface.co';url=f'{base}/api/datasets/kzhou35/mssbench/tree/{REV}?recursive=true&limit=1000';response=requests.get(url,timeout=30);response.raise_for_status();assert 'next' not in response.links;items=response.json();files=[x for x in items if x['type']=='file' and (x['path']=='combined.json' or x['path'].startswith(('chat/','embodied/')))];assert len(files)==753
 (CACHE/'official-tree.json').write_bytes(response.content);raw=CACHE/'raw';raw.mkdir(exist_ok=True);lock=threading.Lock();completed=[]
 def one(item):
  name=item['path'];p=raw/name;p.parent.mkdir(parents=True,exist_ok=True);expected=item.get('lfs',{}).get('oid')
  def valid(data):return len(data)==item['size'] and (digest(data)==expected if expected else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['oid'])
  if not p.exists() or not valid(p.read_bytes()):
   for attempt in range(4):
    try:
     q=requests.get(f'{base}/datasets/kzhou35/mssbench/resolve/{REV}/{name}?download=true',timeout=(10,60));q.raise_for_status();assert valid(q.content),('hash/size',name);tmp=p.with_suffix(p.suffix+'.part');tmp.write_bytes(q.content);tmp.replace(p);break
    except Exception:
     if attempt==3:raise
     time.sleep(1+attempt*2)
  if name!='combined.json':
   from PIL import Image
   with Image.open(p) as im:im.verify()
  with lock:
   completed.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p.read_bytes()),'official_lfs_sha256':expected,'official_git_blob':item['oid']})
   if len(completed)%50==0:print('OFFICIAL_IMAGES_VERIFIED',len(completed),'/',len(files),flush=True)
 with ThreadPoolExecutor(8) as pool:list(pool.map(one,files))
 (CACHE/'acquisition.json').write_text(json.dumps({'dataset':'kzhou35/mssbench','revision':REV,'official_tree_sha256':digest(response.content),'public_ungated':True,'files':sorted(completed,key=lambda x:x['path'])},indent=2)+'\n');print('PUBLIC_MSSBENCH_ACQUIRED',len(completed),flush=True)
if __name__=='__main__':main()
