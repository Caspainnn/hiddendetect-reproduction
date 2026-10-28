"""Only pinned ungated ToxicChat train; preserve immutable acquisition bytes."""
import hashlib
import json
from pathlib import Path
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    revision='29df8e4dba60e1f4af4b4075c0705c5b313548a8'
    url='https://huggingface.co/datasets/lmsys/toxic-chat/resolve/'+revision+'/data/1123/toxic-chat_annotation_train.csv'
    target=ROOT/'data/signal-selection/raw/lmsys__toxic-chat/data/1123/toxic-chat_annotation_train.csv'
    data=target.read_bytes() if target.exists() else urllib.request.urlopen(url,timeout=90).read()
    assert len(data)==7134932
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()=='dbc99e1b2648e53636e9c5b6d9f60f8964ab9682'
    target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
    record={'repo':'lmsys/toxic-chat','revision':revision,'url':url,'relative_path':str(target.relative_to(ROOT)).replace('\\','/'),
        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':'dbc99e1b2648e53636e9c5b6d9f60f8964ab9682',
        'license':'cc-by-nc-4.0','scope':'safe-source calibration only, human_annotation=true and toxicity0',
        'excluded_gated_source':'allenai/wildjailbreak, auto gate not accepted'}
    path=ROOT/'logs/fusion-extra-data.json'
    if path.exists(): assert json.loads(path.read_text())==record
    else: path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PINNED_EXTRA_DATA_VERIFIED',len(data),record['sha256'])
