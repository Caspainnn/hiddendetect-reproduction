import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    meta=json.loads((ROOT/'cache/or-source-metadata.json').read_text());info=meta['info'];assert not info.get('gated')
    assert info.get('cardData',{}).get('license')=='cc-by-4.0'
    revision=info['sha'];repo=info['id'];tree={r['path']:r for r in meta['tree']}
    target=ROOT/'data/oc-confirmation/raw';target.mkdir(parents=True,exist_ok=True);report=[]
    for name in ['README.md','or-bench-80k.csv','or-bench-toxic.csv']:
        item=tree[name];url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
        path=target/name
        if path.exists():payload=path.read_bytes()
        else:
            payload=urllib.request.urlopen(url,timeout=120).read();path.write_bytes(payload)
        assert len(payload)==item['size']
        sha=hashlib.sha256(payload).hexdigest()
        if item.get('lfs'):assert sha==item['lfs']['oid']
        else:assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==item['oid']
        report.append({'file':name,'bytes':len(payload),'sha256':sha,'revision':revision,'repo':repo,'url':url,'license':'cc-by-4.0'})
    out=ROOT/'results/oc-confirmation';out.mkdir(parents=True,exist_ok=True)
    p=out/'acquisition.json'
    if p.exists():assert json.loads(p.read_text())==report
    else:p.write_text(json.dumps(report,indent=2)+'\n')
    print('OR_PUBLIC_FILES_PINNED',revision,[(r['file'],r['bytes']) for r in report],flush=True)
if __name__=='__main__':main()
