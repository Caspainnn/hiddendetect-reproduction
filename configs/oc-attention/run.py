import subprocess,sys,time
from core import ROOT,OUT,write

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for stage in ['check','prepare','train','evaluate','verify','report']:
        write(OUT/'status.json',{'state':'running','stage':stage,'time':time.time()})
        result=subprocess.run([sys.executable,'-u',f'configs/oc-attention/{stage}.py'],cwd=ROOT)
        if result.returncode:
            write(OUT/'status.json',{'state':'failed','stage':stage,'code':result.returncode,'time':time.time()});return result.returncode
    write(OUT/'status.json',{'state':'completed','time':time.time()});return 0
if __name__=='__main__':sys.exit(main())
