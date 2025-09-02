#重新写一下，把之前的程序拆了

import time
from Appendix import *
from SaveLoad import *
import sys

#初筛个数
k=15
b=5

Y=load_variable('GSE167000_Counts'+'_Y.pickle')
X=load_variable('GSE167000_Counts'+'_X.pickle')
zhibiao=load_variable('GSE167000_Counts'+'_zhibiao.pickle')


# 暂存，用于恢复
temp = sys.stdout

# 把输出重定向到文件
f = open('8.GSE167000_Counts.log', 'w')

sys.stdout = f

X0 = np.array(X)
# X0 = X0[:,:100]
# X=X0.tolist()
Y0 = np.array(Y)


results_t={}
# chi2 要求非负
start_time = time.time()
X_minmax = MinMaxScaler().fit_transform(X0)
end_time = time.time()
t = end_time - start_time 

results = {}

# (1) 卡方
start_time = time.time()
chi2_selector = SelectKBest(chi2, k=min(k, X0.shape[1])).fit(X_minmax, Y0)
results["chi2"] = chi2_selector.get_support(indices=True).tolist()
end_time = time.time()
elapsed_time = end_time - start_time 
results_t["chi2"]=elapsed_time+t

# (2) 方差分析 F-score
start_time = time.time()
f_selector = SelectKBest(f_classif, k=min(k, X0.shape[1])).fit(X0, Y0)
results["anova"] = f_selector.get_support(indices=True).tolist()
end_time = time.time()
elapsed_time = end_time - start_time 
results_t["anova"]=elapsed_time+t

# (3) 互信息
start_time = time.time()
mi_selector = SelectKBest(mutual_info_classif, k=min(k, X0.shape[1])).fit(X0, Y0)
results["mutual_info"] = mi_selector.get_support(indices=True).tolist()
end_time = time.time()
elapsed_time = end_time - start_time 
results_t["mutual_info"]=elapsed_time+t

# # (4) 不做筛选，保留所有
# start_time = time.time()
# results["all"] = list(range(X0.shape[1]))
# end_time = time.time()
# elapsed_time = end_time - start_time 
# results_t["all"]=elapsed_time+t

for fs_name, selected_idx in results.items():

    print(f"\n>>> 特征选择方法: {fs_name}, 特征数: {len(selected_idx)}=====================================")


    start_time = time.time()
    house,repeat=newmiddle(X,Y,selected_idx)
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

    print('total time~~~~~~~~~~~',elapsed_time+results_t[fs_name])

f.close()