import subprocess
import os

scripts = ['1.GSE152075.py','1.GSE156063.py','1.GSE157103_ec.py','1.GSE157103_tpm.py','1.GSE161731_xpr.py','1.GSE167000_Counts.py','1.GSE167000_FPKMs.py','1.GSE179277.py','1.GSE188678.py',]

for script in scripts:
    try:
        subprocess.run(['python', script])
    except:
        continue
