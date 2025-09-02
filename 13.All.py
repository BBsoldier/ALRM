import subprocess
import os

scripts = ['13.GSE152075.py','13.GSE156063.py','13.GSE157103_ec.py','13.GSE157103_tpm.py','13.GSE161731_xpr.py','13.GSE167000_Counts.py','13.GSE167000_FPKMs.py','13.GSE179277.py','13.GSE188678.py',]

for script in scripts:
    try:
        subprocess.run(['python', script])
    except:
        continue
