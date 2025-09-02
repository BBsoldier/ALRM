#重新写一下，把之前的程序拆了

import time
from Appendix import *
from SaveLoad import *
import sys

#初筛个数
k=15
b=5

Y=load_variable('GSE161731_xpr'+'_Y.pickle')
X=load_variable('GSE161731_xpr'+'_X.pickle')
zhibiao=load_variable('GSE161731_xpr'+'_zhibiao.pickle')


# 暂存，用于恢复
temp = sys.stdout

# 把输出重定向到文件
f = open('5.GSE161731_xpr.log', 'w')

sys.stdout = f

X0 = np.array(X)
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

# (4) 不做筛选，保留所有
start_time = time.time()
results["all"] = list(range(X0.shape[1]))
end_time = time.time()
elapsed_time = end_time - start_time 
results_t["all"]=elapsed_time+t

models = classification_models()

for fs_name, selected_idx in results.items():
    X_sel = X0[:, selected_idx]  # 取出被选中的列
    results[fs_name] = {}

    print(f"\n>>> 特征选择方法: {fs_name}, 特征数: {len(selected_idx)}")

    for model_name, model in models.items():
        # 交叉验证
        start_time = time.time()
        scores = cross_val_score(model, X_sel, Y0, cv=b, scoring="accuracy")
        cv_acc = np.mean(scores)

        # resubstitution
        model.fit(X_sel, Y0)
        y_pred = model.predict(X_sel)
        resub_acc = accuracy_score(Y0, y_pred)

        results[fs_name][model_name] = (cv_acc, resub_acc)

        end_time = time.time()
        elapsed_time = end_time - start_time 

        print(f"  {model_name:15s}: CV均值 = {cv_acc:.4f}, Resub = {resub_acc:.4f}, Time = {elapsed_time+results_t[fs_name]:.4f}")


f.close()