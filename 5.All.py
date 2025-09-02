import subprocess
import os

scripts = ['5.GSE152075.py','5.GSE156063.py','5.GSE157103_ec.py','5.GSE157103_tpm.py','5.GSE161731_xpr.py','5.GSE167000_Counts.py','5.GSE167000_FPKMs.py','5.GSE179277.py','5.GSE188678.py',]

for script in scripts:
    try:
        subprocess.run(['python', script])
    except:
        continue
