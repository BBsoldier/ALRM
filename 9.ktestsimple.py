#简单的做一下k折

from Appendix import *
from SaveLoad import *
import sys

# 暂存，用于恢复
temp = sys.stdout

# 把输出重定向到文件
f = open('9.r=10.log', 'w')

#修改middle跑一个整的，就是放大r

sys.stdout = f




print('GSE152075========================================================================================')
Y=load_variable('GSE152075'+'_Y.pickle')
X=load_variable('GSE152075'+'_X.pickle')
zhibiao=load_variable('GSE152075'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds2(X,Y,zhibiao,key,gene,k=5,r=10)

print('GSE156063========================================================================================')
Y=load_variable('GSE156063'+'_Y.pickle')
X=load_variable('GSE156063'+'_X.pickle')
zhibiao=load_variable('GSE156063'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds(X,Y,zhibiao,key,gene,k=5,r=15)

print('GSE157103_ec========================================================================================')
Y=load_variable('GSE157103_ec'+'_Y.pickle')
X=load_variable('GSE157103_ec'+'_X.pickle')
zhibiao=load_variable('GSE157103_ec'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds2(X,Y,zhibiao,key,gene,k=5)

print('GSE157103_tpm========================================================================================')
Y=load_variable('GSE157103_tpm'+'_Y.pickle')
X=load_variable('GSE157103_tpm'+'_X.pickle')
zhibiao=load_variable('GSE157103_tpm'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds2(X,Y,zhibiao,key,gene,k=5)

print('GSE161731_xpr========================================================================================')
Y=load_variable('GSE161731_xpr'+'_Y.pickle')
X=load_variable('GSE161731_xpr'+'_X.pickle')
zhibiao=load_variable('GSE161731_xpr'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds(X,Y,zhibiao,key,gene,k=5)

print('GSE167000_Counts========================================================================================')
Y=load_variable('GSE167000_Counts'+'_Y.pickle')
X=load_variable('GSE167000_Counts'+'_X.pickle')
zhibiao=load_variable('GSE167000_Counts'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds2(X,Y,zhibiao,key,gene,k=5)

print('GSE167000_FPKMs========================================================================================')
Y=load_variable('GSE167000_FPKMs'+'_Y.pickle')
X=load_variable('GSE167000_FPKMs'+'_X.pickle')
zhibiao=load_variable('GSE167000_FPKMs'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds(X,Y,zhibiao,key,gene,k=5)

print('GSE179277========================================================================================')
Y=load_variable('GSE179277'+'_Y.pickle')
X=load_variable('GSE179277'+'_X.pickle')
zhibiao=load_variable('GSE179277'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds(X,Y,zhibiao,key,gene,k=5)

print('GSE188678========================================================================================')
Y=load_variable('GSE188678'+'_Y.pickle')
X=load_variable('GSE188678'+'_X.pickle')
zhibiao=load_variable('GSE188678'+'_zhibiao.pickle')

key=zhibiao[:]
gene=zhibiao[:]
newkfolds2(X,Y,zhibiao,key,gene,k=5)


f.close()