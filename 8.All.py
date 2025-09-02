import subprocess
import os

scripts = ['8.GSE152075.py','8.GSE156063.py','8.GSE157103_ec.py','8.GSE157103_tpm.py','8.GSE161731_xpr.py','8.GSE167000_Counts.py','8.GSE167000_FPKMs.py','8.GSE179277.py','8.GSE188678.py',]

for script in scripts:
    try:
        subprocess.run(['python', script])
    except:
        continue
