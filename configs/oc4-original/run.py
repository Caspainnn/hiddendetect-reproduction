import subprocess,sys,time
from core import *
def main():
    for stage in ['prepare','train','evaluate','verify','report']:
        write(OUT/'status.json',{'state':'running','stage':stage,'time':time.time()})
        p=subprocess.run([sys.executable,'-u',f'configs/oc4-original/{stage}.py'],cwd=ROOT)
        if p.returncode:
            write(OUT/'status.json',{'state':'failed','stage':stage,'code':p.returncode,'time':time.time()});return p.returncode
    write(OUT/'status.json',{'state':'completed','time':time.time()});return 0
if __name__=='__main__':sys.exit(main())
