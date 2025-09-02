#检查数据完整性
from Appendix import *
from SaveLoad import *
import sys
Y=load_variable('GSE188678'+'_Y.pickle')
X=load_variable('GSE188678'+'_X.pickle')
zhibiao=load_variable('GSE188678'+'_zhibiao.pickle')
print('this must be same')
print(len(X))
print(len(Y))
print('this must be same')
print(len(X[0]))
print(len(zhibiao))
print('=============================')
print('1:',sum(Y))
print('0:',len(Y)-sum(Y))
print('try every a little')
print(X[0][:10])
print(zhibiao[:10])
# print(Y)

# #there are lots of X is not float
# XX=[[float(j) for j in i] for i in X]
# print(XX[0][:10])
# save_variable(XX,'GSE188678'+'_X.pickle')