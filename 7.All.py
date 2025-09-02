import subprocess
import os

scripts = ['7.GSE152075.py','7.GSE156063.py','7.GSE157103_ec.py','7.GSE157103_tpm.py','7.GSE161731_xpr.py','7.GSE167000_Counts.py','7.GSE167000_FPKMs.py','7.GSE179277.py','7.GSE188678.py',]

for script in scripts:
    try:
        subprocess.run(['python', script])
    except:
        continue
