import json,time
from position_core import ROOT,digest,stream_hash,verify_cache

def main():
    directory=ROOT/'cache/positions';target=directory/'seal.json'
    if target.exists():verify_cache();print('EXISTING_SEAL_VERIFIED');return
    began=time.monotonic();m=json.loads((directory/'metadata.json').read_text());cp=json.loads((directory/'checkpoint.json').read_text())
    assert cp['next_index']==m['count'] and digest(ROOT/'configs/positions/extract.py')==m['extractor_sha256']
    from extract import contract
    assert m['readout_contract_sha256']==contract()
    seal={'manifest_sha256':m['manifest_sha256'],'count':m['count'],'metadata_sha256':digest(directory/'metadata.json'),
        'checkpoint_sha256':digest(directory/'checkpoint.json'),'P_sha256':stream_hash(directory/'P.npy'),'actual_artifacts':[]}
    for item in json.loads((ROOT/'configs/artifacts-lock.json').read_text())['files']:
        path=ROOT/item['target'];assert path.stat().st_size==item['bytes'] and stream_hash(path)==item['sha256']
        seal['actual_artifacts'].append(item)
    seal['seconds']=time.monotonic()-began;target.write_text(json.dumps(seal,indent=2)+'\n')
    (ROOT/'logs/positions-cache-seal.json').write_bytes(target.read_bytes());print('CACHE_AND_MODEL_SEALED',seal['seconds'],flush=True)

if __name__=='__main__':main()
