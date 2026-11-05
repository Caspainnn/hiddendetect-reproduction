import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    info=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/bench-llm/or-bench',timeout=60))
    assert not info.get('gated') and info.get('cardData',{}).get('license')=='cc-by-4.0'
    repo,revision=info['id'],info['sha']
    tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true',timeout=60))
    p=ROOT/'cache/or-source-metadata.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps({'info':info,'tree':tree},indent=2),encoding='utf-8')
    print('OR_PUBLIC_METADATA',repo,revision,info.get('gated'),info['cardData']['license'])
if __name__=='__main__':main()
