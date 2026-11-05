from core import *
from extract import Extractor
def main():
 start=time.monotonic();rows,private,roles,y=load()
 for item in json.loads((ORIGIN/'configs/artifacts-lock.json').read_text())['files']:
  p=ORIGIN/item['target'];assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256']
 x=Extractor();proof=x.smoke([private[int(i)] for i in roles['train'][:2]]);print('REAL_IMAGE_F_AND_ATTENTION_SMOKE_PASS',proof,flush=True);i=int(roles['train'][0]);signals,vision,pools,meta,description=x.row(private[i]);assert signals.shape==(9,14) and vision.shape==(1024,) and pools.shape==(14,3,4096);np.savez(CACHE/'preflight-row.npz',signals=signals,vision=vision,pools=pools);write(CACHE/'preflight-description.json',description);proof['all_five_interfaces']=True;proof['sample_metadata']=meta;proof['counts']=x.counts;proof['seconds_by_operation']=x.seconds;x.close();proof['seconds']=time.monotonic()-start;write(OUT/'preflight.json',proof);print('EI_IM1_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
