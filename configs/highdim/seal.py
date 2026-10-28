"""Streaming byte seals for private cache and actual locked model/tokenizer files."""
import json
import hashlib
import time
from pathlib import Path
from features import ROOT,LAYOUT,digest

def stream_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): result.update(chunk)
    return result.hexdigest()

def verify_seal(directory,seal):
    assert digest(directory/'metadata.json')==seal['metadata_sha256']
    assert digest(directory/'checkpoint.json')==seal['checkpoint_sha256']
    for name,item in seal['arrays'].items():
        path=directory/name
        assert path.stat().st_size==item['bytes'] and stream_hash(path)==item['sha256'],name

def main():
    began=time.monotonic(); directory=ROOT/'cache/highdim'; metadata=json.loads((directory/'metadata.json').read_text())
    assert json.loads((directory/'checkpoint.json').read_text())['next_index']==metadata['count']
    target=directory/'seal.json'
    if target.exists():
        verify_seal(directory,json.loads(target.read_text())); print('EXISTING_SEAL_VERIFIED'); return
    arrays={}
    for name in LAYOUT:
        path=directory/(name+'.npy'); arrays[path.name]={'bytes':path.stat().st_size,'sha256':stream_hash(path)}
        print('CACHE_SEALED',name,flush=True)
    artifacts=[]
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        path=ROOT/item['target']; sha=stream_hash(path)
        assert path.stat().st_size==item['bytes'] and sha==item['sha256'],item['target']
        artifacts.append({'target':item['target'],'bytes':item['bytes'],'sha256':sha})
        print('ARTIFACT_REVERIFIED',item['target'],flush=True)
    seal={'status':'complete','manifest_sha256':metadata['manifest_sha256'],'count':metadata['count'],
        'metadata_sha256':digest(directory/'metadata.json'),'checkpoint_sha256':digest(directory/'checkpoint.json'),
        'arrays':arrays,'actual_artifacts':artifacts,'seconds':time.monotonic()-began,
        'scope':'post-extraction actual byte verification before any supervised fitting'}
    temporary=target.with_suffix('.tmp'); temporary.write_text(json.dumps(seal,indent=2)+'\n'); temporary.replace(target)
    (ROOT/'logs/highdim-cache-seal.json').write_bytes(target.read_bytes())
    print('CACHE_AND_ARTIFACT_SEAL_COMPLETE',seal['seconds'],flush=True)

if __name__=='__main__': main()
