#重新写一下，把之前的程序拆了

import time
from Appendix import *
from SaveLoad import *
import sys

#初筛个数
k=15
b=5

Y=load_variable('GSE152075'+'_Y.pickle')
X=load_variable('GSE152075'+'_X.pickle')
zhibiao=load_variable('GSE152075'+'_zhibiao.pickle')


# 暂存，用于恢复
temp = sys.stdout

# 把输出重定向到文件
f = open('7.GSE152075.log', 'w')

sys.stdout = f

X0 = np.array(X)
Y0 = np.array(Y)


results_t={}
results = {}


start_time = time.time()
scores_sorted=simple_rank_IGR2(X0,Y0)
dna_choose=[i[0] for i in scores_sorted[:k]]
end_time = time.time()
t = end_time - start_time 
results["new2"] = dna_choose
results_t["new2"]=t



models = classification_models()

for fs_name, selected_idx in results.items():
    X_sel = X0[:, selected_idx]  # 取出被选中的列
    results[fs_name] = {}

    print(f"\n>>> 特征选择方法: {fs_name}, 特征数: {len(selected_idx)}")
    print(f"特征: {selected_idx}")

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
        
        # 统计实际使用的特征数
        n_used = get_n_used_features(model, X_sel)

        end_time = time.time()
        elapsed_time = end_time - start_time 

        print(f"  {model_name:15s}: CV均值 = {cv_acc:.4f}, Resub = {resub_acc:.4f},n_features_used = {n_used}, Time = {elapsed_time+results_t[fs_name]:.4f}")


f.close()