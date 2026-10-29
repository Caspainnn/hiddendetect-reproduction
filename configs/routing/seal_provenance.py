"""Bind completed extraction to its actual historical source commit before fitting."""
import json,hashlib,subprocess
from routing_core import ROOT,write,validate_metadata,verify_seal

def main():
    p=ROOT/'logs/routing-cache-seal.json';seal=json.loads(p.read_text());commit='85b2c51'
    meta=json.loads((ROOT/'cache/routing/metadata.json').read_text());manifest=json.loads((ROOT/'configs/routing/manifest.json').read_text())
    source={short:hashlib.sha256(subprocess.check_output(['git','show',commit+':configs/routing/'+name],cwd=ROOT)).hexdigest() for short,name in [('extractor','extract.py'),('core','routing_core.py')]}
    validate_metadata(meta,seal['state'],manifest,seal['manifest_sha256'],source)
    seal.update(extraction_commit=subprocess.check_output(['git','rev-parse',commit],cwd=ROOT,text=True).strip(),extraction_sources=source,metadata=meta)
    write(p,seal);verify_seal();print('EXTRACTION_PROVENANCE_VERIFIED',seal['extraction_commit'])

if __name__=='__main__':main()
