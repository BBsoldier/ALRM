#big count time
import time
from Appendix import *
from SaveLoad import *
import sys

Y=load_variable('GSE167000_FPKMs'+'_Y.pickle')
X=load_variable('GSE167000_FPKMs'+'_X.pickle')
zhibiao=load_variable('GSE167000_FPKMs'+'_zhibiao.pickle')


# 暂存，用于恢复
temp = sys.stdout

# 把输出重定向到文件
f = open('1.GSE167000_FPKMs.log', 'w')

sys.stdout = f

print('start')
start_time = time.time()


print('pos',sum(Y))
print('neg',len(Y)-sum(Y))

# print('\n')
# print(len(Y))
# print(len(X))
# print(len(zhibiao))
# print(len(X[0]))
house,repeat=newbig(X,Y,zhibiao)
b,y,c,d=newsmall(house,X,Y,zhibiao)
print('repeat',repeat)
print('include ID',d)
print('include gene',c)
print('gene count',len(c))
print('error count',b)
print('accurancy',1-b/len(X))
four=ss(Y,y)
wen=[
    'True Positive Rate(sensitivity)',
    'True Negative Rate(specificity)',
    'False Positive Rate',
    'False Negative Rate',
]
for i in range(len(wen)):
    print(wen[i],':',four[i])


end_time = time.time()
elapsed_time = end_time - start_time 
print(f"代码执行时间: {elapsed_time:.6f} 秒")

try:
    newVenn(house,X,Y,title='1.GSE167000_FPKMs.'+'new')
except:
    print('mmax<2')






print('old school~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
start_time = time.time()


print('pos',sum(Y))
print('neg',len(Y)-sum(Y))

# print('\n')
# print(len(Y))
# print(len(X))
# print(len(zhibiao))
# print(len(X[0]))
house,repeat=big(X,Y,zhibiao)
b,y,c,d=small(house,X,Y,zhibiao,zhibiao,zhibiao)
print('repeat',repeat)
print('include ID',d)
print('include gene',c)
print('gene count',len(c))
print('error count',b)
print('accurancy',1-b/len(X))
four=ss(Y,y)
wen=[
    'True Positive Rate(sensitivity)',
    'True Negative Rate(specificity)',
    'False Positive Rate',
    'False Negative Rate',
]
for i in range(len(wen)):
    print(wen[i],':',four[i])


end_time = time.time()
elapsed_time = end_time - start_time 
print(f"代码执行时间: {elapsed_time:.6f} 秒")

try:
    newVenn(house,X,Y,title='1.GSE167000_FPKMs.'+'old')
except:
    print('mmax<2')


print('ancestor~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
start_time = time.time()


print('pos',sum(Y))
print('neg',len(Y)-sum(Y))

# print('\n')
# print(len(Y))
# print(len(X))
# print(len(zhibiao))
# print(len(X[0]))
house,repeat=big4(X,Y,zhibiao)
b,y,c,d=small(house,X,Y,zhibiao,zhibiao,zhibiao)
print('repeat',repeat)
print('include ID',d)
print('include gene',c)
print('gene count',len(c))
print('error count',b)
print('accurancy',1-b/len(X))
four=ss(Y,y)
wen=[
    'True Positive Rate(sensitivity)',
    'True Negative Rate(specificity)',
    'False Positive Rate',
    'False Negative Rate',
]
for i in range(len(wen)):
    print(wen[i],':',four[i])


end_time = time.time()
elapsed_time = end_time - start_time 
print(f"代码执行时间: {elapsed_time:.6f} 秒")


try:
    Venn4(house,X,Y,zhibiao,title='1.GSE167000_FPKMs.'+'ancestor')
except:
    print('mmax<2')




f.close()