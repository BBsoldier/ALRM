#每一个模型不再去掉属性，这样有重复的可能

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3,venn2
import math

def LR(X,Y):
    # 初始化逻辑回归模型
    log_reg = LogisticRegression()
    # 训练模型
    log_reg.fit(X, Y)
    return log_reg

def count_point(y0,y1):
    p0=0
    for i in range(len(y0)):
        if y0[i]!=y1[i]:
            p0+=1
    return p0

def count_point0(y_pred,survive_flag):
    p0=0
    #总错误数
    for i in range(len(y_pred)):
        p0+=abs(y_pred[i]-survive_flag[i])
    p1=0
    #0错误数
    for i in range(len(survive_flag)):
        if survive_flag[i]==0 and y_pred[i]==1:
            p1+=1
    return p0,p1



def combinations(source: list,n:int)->list:
    '''从一个元素不重复的列表里选出n个元素
    :参数 source:元素不重复的列表
    :参数 n: 要选出的元素数量，正整数，小于等于列表source的长度
    '''
    # 如果n正好等于列表的长度，那么只有一种组合
    # 就是把所有元素都选出来
    if len(source)==n:
        return [source]
    # 如果n是1，那么列表中的每个元素都是一个组合
    if n == 1:
        ans = []
        for i in source:
            ans.append([i])
        return ans
    # 下面处理n小于列表长度的情况
    ans = []
    # 从列表里选n个元素出来，可以理解为先把列表里的第0个元素拿出来放进组合
    # 然后从剩下的元素里选出n-1个
    for each_list in combinations(source[1:],n-1):
        ans.append([source[0]]+each_list)
    # 还可以直接从剩下的元素里选出n个
    for each_list in combinations(source[1:],n):
        ans.append(each_list)
    return ans

def big(X,Y,dna_name):
    #应该是要判断为0判断的最准的max
    #同理判断1的话就要min
    #比middle多加了一个排序
    judge=0
    survive=X[:]
    survive_flag=Y[:]
    house=[]
    r=10
    kmax=3
    mmax=4
    m=0
    repeat=[]
    while judge==0:
        m+=1
        #先排名前100
        dna_point=[]
        for i in range(len(dna_name)):
        # for i in range(100): #test
            # print(i)
            x=[]
            for j in range(len(survive)):
                x.append([survive[j][i]])

            log_reg=LR(x,survive_flag)

            y_pred = log_reg.predict_proba(x)[:, 1]
            fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
            
            # 计算 Youden's J statistic 并找到最佳阈值
            j_scores = tpr - fpr
            best_threshold_index = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_index]
            y_p0= (y_pred  >= best_threshold).astype(int)
            dna_point.append([count_point0(y_p0,survive_flag)[0],dna_name[i]])
        dna_point.sort(reverse=False,key=lambda element: element[0])
        # 如果比p1的话，就有可能找出来的时全是0的
        # print(dna_point[0][0])
        # print(dna_point[1][0])
        dna_choose=[i[1] for i in dna_point[:r]]
        k=0
        judge1=0

        best_group=[]
        while judge1==0:
            
            k+=1
            a=combinations(dna_choose,k)
            print('x amount',k)
            print('model amount',len(a))

            for b in a:
                # model=b
                index=[]
                for i in range(len(dna_name)):
                    if dna_name[i] in b:
                        index.append(i)
                # index.sort(reverse=False)
                b=[dna_name[i] for i in index]
                x=[]
                for i in range(len(survive)):
                    xx=[]
                    for j in index:
                        xx.append(survive[i][j])
                    x.append(xx)
                log_reg=LR(x,survive_flag)
                y_pred = log_reg.predict_proba(x)[:, 1]
                fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
                
                # 计算 Youden's J statistic 并找到最佳阈值
                j_scores = tpr - fpr
                best_threshold_index = np.argmax(j_scores)
                best_threshold = thresholds[best_threshold_index]
                y_p0= (y_pred  >= best_threshold).astype(int)
                p0,p1=count_point0(y_p0,survive_flag)
                best_group.append([p1,p0,b,log_reg,best_threshold,y_p0])
            if k==kmax:
                judge1=1
        best_group.sort(reverse=False,key=lambda element: element[0])
        best_finn=[]
        for single in best_group:
            if single[0]==best_group[0][0]:
                best_finn.append(single)
        best_finn.sort(reverse=False,key=lambda element: element[1])
        
        best_fin=[]
        for single in best_finn:
            # if single[1]==best_group[0][1]:#如果不对劲就再改回来
            if single[1]==best_finn[0][1]:
                best_fin.append(single)
        best_fin.sort(reverse=False,key=lambda element: len(element[2]))
        
        
        repeat.append(len(best_fin))
        house.append(best_fin[0])
        #把剩下的survive输出出来
        y_p0=best_fin[0][5]
        # print(y_p0)
        # print(survive_flag)
        if count_point(y_p0,survive_flag)==0:
            print('nomal end')
            break
        else:
            s=[]
            ss=[]
            for i in range(len(y_p0)):
                if y_p0[i]==0:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
            print('this round kill 0',len(survive_flag)-sum(survive_flag)-(len(ss)-sum(ss)))
            print('this round kill 1',sum(survive_flag)-sum(ss))

            survive=s[:]
            survive_flag=ss[:]
            
            print('next round survive number',len(survive_flag))
            print('next round 1 number',sum(survive_flag))
            print('next round 0 number',len(survive_flag)-sum(survive_flag))
            if sum(survive_flag)==len(survive_flag) or sum(survive_flag)==0 or m==mmax:
                print('end')
                break
            print('\n')
        
    # save_variable(house,'house.pickle')
    # [p1,p0,model,log_reg,best_threshold,y_p0]
    return house,repeat

#把探针和基因对应起来
def opendoor(shuxing,key,gene):
    try:
        index=key.index(shuxing)
        g=gene[index]
        return g
    except:
        return shuxing


def small(house,X,Y,dna_name,key,gene):
    choose_gene=[]
    choose_id=[]
    yy=[]
    for model in house:
        dna=[]
        for i in model[2]:
            c=opendoor(i,key,gene)
            dna.append(c)
            # print('66666',c)
            if c not in choose_gene:
                choose_gene.append(c)
            if i not in choose_id:
                choose_id.append(i)
        print('~~~~~~~~~~~')
        print('Gene Symbol:',dna)
        print('ID:',model[2])
        print('coef',model[3].coef_)
        print('intercept',model[3].intercept_)
        print('threshold',model[4])
        index=[]
        for i in range(len(dna_name)):
            if dna_name[i] in model[2]:
                index.append(i)
        x=[]
        for i in range(len(X)):
            xx=[]
            for j in index:
                xx.append(X[i][j])
            x.append(xx)

        y_pred = model[3].predict_proba(x)[:, 1]
        y_p0= (y_pred  >= model[4]).astype(int)
        yy.append(y_p0)

    y=[]
    for i in range(len(yy[0])):#group1 和 group2
        yyy=[]
        for m in yy:
            yyy.append(m[i])
        y.append(int(max(yyy)))

    return count_point(y,Y),y,choose_gene,choose_id

def ss(y_p0,Y):
    L=len(y_p0)
    TP=0 #正例人数
    TN=0 #正例人数
    FP=0 #正例人数
    FN=0 #正例人数
    for i in range(len(y_p0)):
        if y_p0[i]==1 and Y[i]==1:
            TP+=1
        elif y_p0[i]==1 and Y[i]==0:
            FP+=1
        elif y_p0[i]==0 and Y[i]==1:
            FN+=1
        elif y_p0[i]==0 and Y[i]==0:
            TN+=1
    TPR = TP /(TP + FN)
    #True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）
    #正样本预测结果数 / 正样本实际数
    TNR = TN /(TN + FP)
    #True Negative Rate（真负率 , TNR）或特指度（specificity）
    #负样本预测结果数 / 负样本实际数
    FPR = FP /(FP + TN)
    #False Positive Rate （假正率, FPR）
    #被预测为正的负样本结果数 /负样本实际数
    FNR = FN /(TP + FN)
    # False Negative Rate（假负率 , FNR）
    # 被预测为负的正样本结果数 / 正样本实际数
    four=[TPR,TNR,FPR,FNR]
    return four

# [p1,p0,model,log_reg,best_threshold,y_p0]
def Venn3(house,X,Y,zhibiao,title='Venn3'):
    Set=[]
    SSet=[]
    for model in house[:3]:

        index=[]
        for i in range(len(zhibiao)):
            if zhibiao[i] in model[2]:
                index.append(i)
        x=[]
        for i in range(len(X)):
            xx=[]
            for j in index:
                xx.append(X[i][j])
            x.append(xx)

        y_pred = model[3].predict_proba(x)[:, 1]
        y_p0= (y_pred  >= model[4]).astype(int)

        # print(Y)
        s=[]
        ss=[]
        for i in range(len(y_p0)):
            # print(y_p0[i],Y[i])
            if y_p0[i]==Y[i]:
                s.append(str(i))
                if y_p0[i]==1:
                    ss.append(str(i))
        Set.append(set(s))
        SSet.append(set(ss))
        # print(SSet)
        # print('next model\n')
    # 绘制三个集合的维恩图
    
    try:
        plt.figure(figsize=(8, 8))
        venn = venn3(Set, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$', r'$sm_'+str(3)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(with 0)')
        plt.savefig(title+'_01.png')
        # plt.show()
        plt.close()
    except:
        plt.figure(figsize=(8, 8))
        venn = venn2(Set, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(with 0)')
        plt.savefig(title+'_01.png')
        # plt.show()
        plt.close()

    # 绘制三个集合的维恩图
    try:
        plt.figure(figsize=(8, 8))
        # print(SSet)
        venn = venn3(SSet, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$', r'$sm_'+str(3)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(only 1)')
        plt.savefig(title+'_1.png')
        # plt.show()
        plt.close()
    except:
        plt.figure(figsize=(8, 8))
        # print(SSet)
        venn = venn2(SSet, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(only 1)')
        plt.savefig(title+'_1.png')
        # plt.show()
        plt.close()

#为了使得sm所标记的不同
# [p1,p0,b,log_reg,best_threshold,y_p0]
def Venn4(house,X,Y,zhibiao,title='Venn3',sm=1):
    Set=[]
    SSet=[]
    for model in house[:3]:

        index=[]
        for i in range(len(zhibiao)):
            if zhibiao[i] in model[2]:
                index.append(i)
        x=[]
        for i in range(len(X)):
            xx=[]
            for j in index:
                xx.append(X[i][j])
            x.append(xx)

        y_pred = model[3].predict_proba(x)[:, 1]
        y_p0= (y_pred  >= model[4]).astype(int)

        # print(Y)
        s=[]
        ss=[]
        for i in range(len(y_p0)):
            # print(y_p0[i],Y[i])
            if y_p0[i]==Y[i]:
                s.append(str(i))
                if y_p0[i]==1:
                    ss.append(str(i))
        Set.append(set(s))
        SSet.append(set(ss))
        # print(SSet)
        # print('next model\n')
    # 绘制三个集合的维恩图
    
    try:
        plt.figure(figsize=(8, 8))
        venn = venn3(Set, (r'$sm_'+str(sm)+'$', r'$sm_'+str(sm+1)+'$', r'$sm_'+str(sm+2)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(with 0)')
        plt.savefig(title+'_01.png')
        # plt.show()
        plt.close()
    except:
        plt.figure(figsize=(8, 8))
        venn = venn2(Set, (r'$sm_'+str(sm)+'$', r'$sm_'+str(sm+1)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(with 0)')
        plt.savefig(title+'_01.png')
        # plt.show()
        plt.close()

    # 绘制三个集合的维恩图
    try:
        plt.figure(figsize=(8, 8))
        # print(SSet)
        venn = venn3(SSet, (r'$sm_'+str(sm)+'$', r'$sm_'+str(sm+1)+'$', r'$sm_'+str(sm+2)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(only 1)')
        plt.savefig(title+'_1.png')
        # plt.show()
        plt.close()
    except:
        plt.figure(figsize=(8, 8))
        # print(SSet)
        venn = venn2(SSet, (r'$sm_'+str(sm)+'$', r'$sm_'+str(sm+1)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(16)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(16)
        # plt.title('Venn Diagram of Three Sets(only 1)')
        plt.savefig(title+'_1.png')
        # plt.show()
        plt.close()

#严格复现老算法，所有错误都加入下一轮了
def big4(X,Y,dna_name):
    judge=0
    survive=X[:]
    survive_flag=Y[:]
    house=[]
    r=15
    kmax=3
    mmax=4
    m=0
    repeat=[]
    while judge==0:
        m+=1
        #先排名前100
        dna_point=[]
        for i in range(len(dna_name)):
        # for i in range(100): #test
            # print(i)
            x=[]
            for j in range(len(survive)):
                x.append([survive[j][i]])

            log_reg=LR(x,survive_flag)

            y_pred = log_reg.predict_proba(x)[:, 1]
            fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
            
            # 计算 Youden's J statistic 并找到最佳阈值
            j_scores = tpr - fpr
            best_threshold_index = np.argmax(j_scores)
            best_threshold = thresholds[best_threshold_index]
            y_p0= (y_pred  >= best_threshold).astype(int)
            dna_point.append([count_point0(y_p0,survive_flag)[0],dna_name[i]])
        dna_point.sort(reverse=False,key=lambda element: element[0])
        # 如果比p1的话，就有可能找出来的时全是0的
        # print(dna_point[0][0])
        # print(dna_point[1][0])
        dna_choose=[i[1] for i in dna_point[:r]]
        k=0
        judge1=0

        best_group=[]
        while judge1==0:
            
            k+=1
            a=combinations(dna_choose,k)
            print('x amount',k)
            print('model amount',len(a))

            for b in a:
                # model=b
                index=[]
                for i in range(len(dna_name)):
                    if dna_name[i] in b:
                        index.append(i)
                # index.sort(reverse=False)
                x=[]
                b=[dna_name[i] for i in index]
                for i in range(len(survive)):
                    xx=[]
                    for j in index:
                        xx.append(survive[i][j])
                    x.append(xx)
                log_reg=LR(x,survive_flag)
                y_pred = log_reg.predict_proba(x)[:, 1]
                fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
                
                # 计算 Youden's J statistic 并找到最佳阈值
                j_scores = tpr - fpr
                best_threshold_index = np.argmax(j_scores)
                best_threshold = thresholds[best_threshold_index]
                y_p0= (y_pred  >= best_threshold).astype(int)
                p0,p1=count_point0(y_p0,survive_flag)
                best_group.append([p1,p0,b,log_reg,best_threshold,y_p0])
            if k==kmax:
                judge1=1
        best_group.sort(reverse=False,key=lambda element: element[0])
        best_finn=[]
        for single in best_group:
            if single[0]==best_group[0][0]:
                best_finn.append(single)
        best_finn.sort(reverse=False,key=lambda element: element[1])
        
        best_fin=[]
        for single in best_finn:
            if single[1]>=best_group[0][1]:#如果不对劲就再改回来
            # if single[1]==best_finn[0][1]:
                best_fin.append(single)
        best_fin.sort(reverse=False,key=lambda element: len(element[2]))
        
        
        repeat.append(len(best_fin))
        house.append(best_fin[0])
        #把剩下的survive输出出来
        y_p0=best_fin[0][5]
        # print(y_p0)
        # print(survive_flag)
        if count_point(y_p0,survive_flag)==0:
            print('nomal end')
            break
        else:
            s=[]
            ss=[]
            for i in range(len(y_p0)):
                if not y_p0[i]==survive_flag[i]:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
                elif y_p0[i]==0 and survive_flag[i]==0:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
            print('this round kill 0',len(survive_flag)-sum(survive_flag)-(len(ss)-sum(ss)))
            print('this round kill 1',sum(survive_flag)-sum(ss))

            survive=s[:]
            survive_flag=ss[:]
            
            print('next round survive number',len(survive_flag))
            print('next round 1 number',sum(survive_flag))
            print('next round 0 number',len(survive_flag)-sum(survive_flag))
            if sum(survive_flag)==len(survive_flag) or sum(survive_flag)==0 or m==mmax:
                print('end')
                break
            print('\n')
        
    # save_variable(house,'house.pickle')
    # [p1,p0,model,log_reg,best_threshold,y_p0]
    return house,repeat

#老算法0.5
def big5(X,Y,dna_name):
    judge=0
    survive=X[:]
    survive_flag=Y[:]
    house=[]
    r=15
    kmax=3
    mmax=4
    m=0
    repeat=[]
    while judge==0:
        m+=1
        #先排名前100
        dna_point=[]
        for i in range(len(dna_name)):
        # for i in range(100): #test
            # print(i)
            x=[]
            for j in range(len(survive)):
                x.append([survive[j][i]])

            log_reg=LR(x,survive_flag)

            y_pred = log_reg.predict_proba(x)[:, 1]
            best_threshold = 0.5
            y_p0= (y_pred  >= best_threshold).astype(int)
            dna_point.append([count_point0(y_p0,survive_flag)[0],dna_name[i]])
        dna_point.sort(reverse=False,key=lambda element: element[0])
        # 如果比p1的话，就有可能找出来的时全是0的
        # print(dna_point[0][0])
        # print(dna_point[1][0])
        dna_choose=[i[1] for i in dna_point[:r]]
        k=0
        judge1=0

        best_group=[]
        while judge1==0:
            
            k+=1
            a=combinations(dna_choose,k)
            print('x amount',k)
            print('model amount',len(a))

            for b in a:
                # model=b
                index=[]
                for i in range(len(dna_name)):
                    if dna_name[i] in b:
                        index.append(i)
                # index.sort(reverse=False)
                b=[dna_name[i] for i in index]
                x=[]
                for i in range(len(survive)):
                    xx=[]
                    for j in index:
                        xx.append(survive[i][j])
                    x.append(xx)
                log_reg=LR(x,survive_flag)
                y_pred = log_reg.predict_proba(x)[:, 1]
                
                best_threshold = 0.5
                y_p0= (y_pred  >= best_threshold).astype(int)
                p0,p1=count_point0(y_p0,survive_flag)
                best_group.append([p1,p0,b,log_reg,best_threshold,y_p0])
            if k==kmax:
                judge1=1
        best_group.sort(reverse=False,key=lambda element: element[0])
        best_finn=[]
        for single in best_group:
            if single[0]==best_group[0][0]:
                best_finn.append(single)
        best_finn.sort(reverse=False,key=lambda element: element[1])
        
        best_fin=[]
        for single in best_finn:
            if single[1]>=best_group[0][1]:#如果不对劲就再改回来
            # if single[1]==best_finn[0][1]:
                best_fin.append(single)
        best_fin.sort(reverse=False,key=lambda element: len(element[2]))
        
        
        repeat.append(len(best_fin))
        house.append(best_fin[0])
        #把剩下的survive输出出来
        y_p0=best_fin[0][5]
        # print(y_p0)
        # print(survive_flag)
        if count_point(y_p0,survive_flag)==0:
            print('nomal end')
            break
        else:
            s=[]
            ss=[]
            for i in range(len(y_p0)):
                if not y_p0[i]==survive_flag[i]:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
                elif y_p0[i]==0 and survive_flag[i]==0:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
            print('this round kill 0',len(survive_flag)-sum(survive_flag)-(len(ss)-sum(ss)))
            print('this round kill 1',sum(survive_flag)-sum(ss))

            survive=s[:]
            survive_flag=ss[:]
            
            print('next round survive number',len(survive_flag))
            print('next round 1 number',sum(survive_flag))
            print('next round 0 number',len(survive_flag)-sum(survive_flag))
            if sum(survive_flag)==len(survive_flag) or sum(survive_flag)==0 or m==mmax:
                print('end')
                break
            print('\n')
        
    # save_variable(house,'house.pickle')
    # [p1,p0,model,log_reg,best_threshold,y_p0]
    return house,repeat



def knum(zongshu,k):
    extra=zongshu%k
    stard=zongshu//k
    a=[stard]*(k-extra)+[stard+1]*extra
    return a

#k折交叉验证
def kfolds(X,Y,dna_name,key,gene,k=5):
    #正式程序开始

    #这里我想设置一个调整数据顺序的东西，因为有的数据并不是0011这么规矩排好的，有可能是0101这种
    # XY=[]
    # for i,j in X,Y:
    #     XY.append(i+j)
    
    index1=0
    index0=0
    for i in range(len(Y)):
        if Y[i]==1:
            index1=i
            break
    for i in range(len(Y)):
        if Y[i]==0:
            index0=i
            break
    num1=index0
    num0=len(Y)-index0
    a1=knum(num1,k)
    a0=knum(num0,k)
    error=[]
    # print(index1,index0)
    # print(a1)
    # print(a0)
    y_p0=[]
    y_0=[]


    y_p=[]
    for i in range(len(a1)):
        #测定index范围
        choose=[]#记录该人是否被选中训练
        for j in range(len(Y)):
            if j >=index1 and j<index1+a1[i]:
                choose.append(1)
            elif j >=index0 and j<index0+a0[i]:
                choose.append(1)
            else:
                choose.append(0)
        # print(choose)
        index1+=a1[i]
        index0+=a0[i]

        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        for j in range(len(choose)):
            if choose[j]==0:
                y_train.append(Y[j])
                x_train.append(X[j])
            elif choose[j]==1:
                y_test.append(Y[j])
                x_test.append(X[j])
        
        house,repeat=big4(x_train,y_train,dna_name)
        b,y,c,d=small(house,x_test,y_test,dna_name,key,gene)
        four=ss(y_test,y)
        wen=[
            'True Positive Rate(sensitivity)',
            'True Negative Rate(specificity)',
            'False Positive Rate',
            'False Negative Rate',
        ]
        for i in range(len(wen)):
            print(wen[i],':',four[i])
        error.append(b)
        print('test',b)
        y_p0+=y
        y_0+=y_test
        b,y,c,d=small(house,x_train,y_train,dna_name,key,gene)
        print('self',c)
        print('repeat',repeat)
        print('include ID',d)
        print('include gene',c)
        print('gene count',len(c))
        print('error count',b)
        print('accurancy',1-b/len(X))
        

        
        
        print('!!!!!!!!!!!!!!!!!!!')
        print('\n')
    print('\nfinal')
    print(error)
    print(sum(error))
    print('accuracy',1-sum(error)/len(Y))
    try:
        print(ss(y_p0,y_0))
    except:
        print('TNR = TN /(TN + FP): division by zero')



#把其他数据集的基因池折合到本数据集
def findkey(choose_gene,zhibiao):
    #先提出真正的基因
    origin_gene=[]
    for i in choose_gene:
        a=i.split('_')
        if not '/' in a[0]:
            origin_gene.append(a[0].strip())
        else:
            b=a[0].split('/')
            origin_gene.append(b[0])
    change_gene=[]
    for i in zhibiao:
        a=i.split('_')
        if not '/' in a[0]:
            if a[0].strip() in origin_gene:
                change_gene.append(i)
        else:
            b=a[0].split('/')
            if b[0].strip() in origin_gene:
                change_gene.append(i)
    return origin_gene,change_gene

#把其他数据集的基因池折合到本数据集,双阈值版
def findkey1(choose_gene,zhibiao):
    #先提出真正的基因
    origin_gene=[]
    for i in choose_gene:
        a=i.split('~')
        if not '/' in a[0]:
            origin_gene.append(a[0].strip())
        else:
            b=a[0].split('/')
            for j in b:
                if j !='':
                    origin_gene.append(j.strip())
    change_gene=[]
    for i in origin_gene:
        for j in zhibiao:
            if i in j and j not in change_gene:
                change_gene.append(j)
    return origin_gene,change_gene

#获取列表中重复的元素
def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)


import random
def mix_survive(Y,X):
    YY=[]
    XX=[]
    YX=[]
    for i in range(len(Y)):
        YX.append([Y[i],X[i]])
    # print(YX[0])
    random.shuffle(YX)
    # print(YX[0])
    YX.sort(key=lambda x:x[0],reverse=True)
    # print(YX[0])
    for i in range(len(YX)):
        YY.append(YX[i][0])
        XX.append(YX[i][1])
    return YY,XX



def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

# 计算信息增益
def information_gain(X_col, y):
    total_entropy = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_y = y[X_col == v]
        weighted_entropy += (c / len(y)) * entropy(subset_y)
    return total_entropy - weighted_entropy

# 计算分裂信息
def split_info(X_col):
    values, counts = np.unique(X_col, return_counts=True)
    probs = counts / len(X_col)
    return -np.sum(probs * np.log2(probs + 1e-9))

# 信息增益比
def information_gain_ratio(X_col, y):
    ig = information_gain(X_col, y)
    si = split_info(X_col)
    return ig / (si + 1e-9)

def compare(x):
    if math.isnan(x):
        return -float('inf')  # 将 nan 值替换为无穷大值
    else:
        return x

# 主函数：对二维列表 X (n*p) 进行IGR特征筛选
# 这个还是用排名去分割
def simple_rank_IGR(X, y):
    X = np.array(X)  # 转为numpy数组，形状 (n, p)
    y = np.array(y)
    n, p = X.shape
    scores = []
    l=sum(y)
    for j in range(p):
        # 对每个基因（列）做中位数二值化

        # 在这儿跳跃！
        # 这样也就解决了不平衡问题
        # 且两种01情况都进去比了
        t=list(X[:, j])
        t.sort()
        threshold = t[l]
        X_discrete = (X[:, j] > threshold).astype(int)
        igr = information_gain_ratio(X_discrete, y)
        scores.append([j, compare(igr)])  # 记录基因编号和IGR
        threshold = t[-l-1]
        X_discrete = (X[:, j] > threshold).astype(int)
        igr = information_gain_ratio(X_discrete, y)
        scores.append([j, compare(igr)])  # 记录基因编号和IGR

    # 按 IGR 值排序，取前 r 个
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted

# 这个直接用约旦阈值分割
def simple_rank_IGR2(X, y):
    X = np.array(X)  # 转为numpy数组，形状 (n, p)
    y = np.array(y)
    n, p = X.shape
    scores = []
    # l=sum(y)
    for j in range(p):
        #01无所谓，只要分成了两类就行
        xx=[[i] for i in X[:, j]]
        log_reg=LR(xx,y)
        y_pred = log_reg.predict_proba(xx)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        j_scores = tpr - fpr
        best_threshold_index = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_index]
        y_p0= (y_pred  >= best_threshold).astype(int)
        igr = information_gain_ratio(y_p0, y)
        # p0=count_point(y_p0,y)
        scores.append([j, compare(igr)]) 
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores_sorted

#随手一筐
def easy_catch(X, y):
    X = np.array(X)  # 转为numpy数组，形状 (n, p)
    y = np.array(y)
    n, p = X.shape
    scores = []
    l=sum(y)
    for j in range(p):
        # 对每个基因（列）做中位数二值化
        # print(j,'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # 在这儿跳跃！
        # 这样也就解决了不平衡问题
        # 且两种01情况都进去比了
        t=list(X[:, j])
        t.sort()
        threshold = t[l]
        X_discrete = (X[:, j] > threshold).astype(int)
        p0=count_point(X_discrete, y)
        scores.append([j,p0])  # 记录基因编号和IGR
        threshold = t[-l-1]
        X_discrete = (X[:, j] > threshold).astype(int)
        p0=count_point(X_discrete, y)
        scores.append([j,p0])  # 记录基因编号和IGR

    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=False)
    return scores_sorted

#换初筛方式
def newbig(X,Y,dna_name,r=15):
    #应该是要判断为0判断的最准的max
    #同理判断1的话就要min
    #比middle多加了一个排序
    judge=0
    survive=X[:]
    survive_flag=Y[:]
    house=[]

    kmax=3
    mmax=4
    m=0
    repeat=[]
    while judge==0:
        m+=1
        #先排名前100
        top_genes = simple_rank_IGR(survive, survive_flag)
        # top_genes = easy_catch(survive, survive_flag)
        dna_choose=[i[0] for i in top_genes[:r]]
        k=0
        judge1=0

        best_group=[]
        while judge1==0:
            
            k+=1
            a=combinations(dna_choose,k)
            print('x amount',k)
            print('model amount',len(a))

            for b in a:
                b.sort(reverse=False)
                x=[]
                for i in range(len(survive)):
                    xx=[]
                    for j in b:
                        xx.append(survive[i][j])
                    x.append(xx)
                log_reg=LR(x,survive_flag)
                y_pred = log_reg.predict_proba(x)[:, 1]
                fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
                
                # 计算 Youden's J statistic 并找到最佳阈值
                j_scores = tpr - fpr
                best_threshold_index = np.argmax(j_scores)
                best_threshold = thresholds[best_threshold_index]
                y_p0= (y_pred  >= best_threshold).astype(int)
                p0,p1=count_point0(y_p0,survive_flag)
                best_group.append([p1,p0,b,log_reg,best_threshold,y_p0])
            if k==kmax:
                judge1=1
        best_group.sort(reverse=False,key=lambda element: element[0])
        best_finn=[]
        for single in best_group:
            if single[0]==best_group[0][0]:
                best_finn.append(single)
        best_finn.sort(reverse=False,key=lambda element: element[1])
        
        best_fin=[]
        for single in best_finn:
            # if single[1]==best_group[0][1]:#如果不对劲就再改回来
            if single[1]==best_finn[0][1]:
                best_fin.append(single)
        best_fin.sort(reverse=False,key=lambda element: len(element[2]))
        
        
        repeat.append(len(best_fin))
        house.append(best_fin[0])
        #把剩下的survive输出出来
        y_p0=best_fin[0][5]
        # print(y_p0)
        # print(survive_flag)
        if count_point(y_p0,survive_flag)==0:
            print('nomal end')
            break
        else:
            s=[]
            ss=[]
            for i in range(len(y_p0)):
                if y_p0[i]==0:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
            print('this round kill 0',len(survive_flag)-sum(survive_flag)-(len(ss)-sum(ss)))
            print('this round kill 1',sum(survive_flag)-sum(ss))

            survive=s[:]
            survive_flag=ss[:]
            
            print('next round survive number',len(survive_flag))
            print('next round 1 number',sum(survive_flag))
            print('next round 0 number',len(survive_flag)-sum(survive_flag))
            if sum(survive_flag)==len(survive_flag) or sum(survive_flag)==0 or m==mmax:
                print('end')
                break
            print('\n')
        
    # save_variable(house,'house.pickle')
    # [p1,p0,model,log_reg,best_threshold,y_p0]
    return house,repeat

def newbig2(X,Y,dna_name,r=15):
    #应该是要判断为0判断的最准的max
    #同理判断1的话就要min
    #比middle多加了一个排序
    judge=0
    survive=X[:]
    survive_flag=Y[:]
    house=[]

    kmax=3
    mmax=4
    m=0
    repeat=[]
    while judge==0:
        m+=1
        #先排名前100
        top_genes = simple_rank_IGR2(survive, survive_flag)
        # top_genes = easy_catch(survive, survive_flag)
        dna_choose=[i[0] for i in top_genes[:r]]
        k=0
        judge1=0

        best_group=[]
        while judge1==0:
            
            k+=1
            a=combinations(dna_choose,k)
            print('x amount',k)
            print('model amount',len(a))

            for b in a:
                b.sort(reverse=False)
                x=[]
                for i in range(len(survive)):
                    xx=[]
                    for j in b:
                        xx.append(survive[i][j])
                    x.append(xx)
                log_reg=LR(x,survive_flag)
                y_pred = log_reg.predict_proba(x)[:, 1]
                fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
                
                # 计算 Youden's J statistic 并找到最佳阈值
                j_scores = tpr - fpr
                best_threshold_index = np.argmax(j_scores)
                best_threshold = thresholds[best_threshold_index]
                y_p0= (y_pred  >= best_threshold).astype(int)
                p0,p1=count_point0(y_p0,survive_flag)
                best_group.append([p1,p0,b,log_reg,best_threshold,y_p0])
            if k==kmax:
                judge1=1
        best_group.sort(reverse=False,key=lambda element: element[0])
        best_finn=[]
        for single in best_group:
            if single[0]==best_group[0][0]:
                best_finn.append(single)
        best_finn.sort(reverse=False,key=lambda element: element[1])
        
        best_fin=[]
        for single in best_finn:
            # if single[1]==best_group[0][1]:#如果不对劲就再改回来
            if single[1]==best_finn[0][1]:
                best_fin.append(single)
        best_fin.sort(reverse=False,key=lambda element: len(element[2]))
        
        
        repeat.append(len(best_fin))
        house.append(best_fin[0])
        #把剩下的survive输出出来
        y_p0=best_fin[0][5]
        # print(y_p0)
        # print(survive_flag)
        if count_point(y_p0,survive_flag)==0:
            print('nomal end')
            break
        else:
            s=[]
            ss=[]
            for i in range(len(y_p0)):
                if y_p0[i]==0:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
            print('this round kill 0',len(survive_flag)-sum(survive_flag)-(len(ss)-sum(ss)))
            print('this round kill 1',sum(survive_flag)-sum(ss))

            survive=s[:]
            survive_flag=ss[:]
            
            print('next round survive number',len(survive_flag))
            print('next round 1 number',sum(survive_flag))
            print('next round 0 number',len(survive_flag)-sum(survive_flag))
            if sum(survive_flag)==len(survive_flag) or sum(survive_flag)==0 or m==mmax:
                print('end')
                break
            print('\n')
        
    # save_variable(house,'house.pickle')
    # [p1,p0,model,log_reg,best_threshold,y_p0]
    return house,repeat


#新算法
def newmiddle(X,Y,index_choose):
    #应该是要判断为0判断的最准的max
    #同理判断1的话就要min
    #比middle多加了一个排序
    judge=0
    survive=X[:]
    survive_flag=Y[:]
    house=[]
    kmax=min(3,len(index_choose))
    mmax=4
    m=0
    repeat=[]
    while judge==0:
        m+=1

        k=0
        judge1=0

        best_group=[]
        while judge1==0:
            
            k+=1
            a=combinations(index_choose,k)
            print('x amount',k)
            print('model amount',len(a))
            for b in a:

                X0 = np.array(survive)
                x= X0[:,b] 
                # x=[]
                # for i in range(len(survive)):
                #     xx=[]
                #     for j in index_choose:
                #         xx.append(survive[i][j])
                #     x.append(xx)
                log_reg=LR(x,survive_flag)
                y_pred = log_reg.predict_proba(x)[:, 1]
                fpr, tpr, thresholds = roc_curve(survive_flag, y_pred)
                
                # 计算 Youden's J statistic 并找到最佳阈值
                j_scores = tpr - fpr
                best_threshold_index = np.argmax(j_scores)
                best_threshold = thresholds[best_threshold_index]
                y_p0= (y_pred  >= best_threshold).astype(int)
                p0,p1=count_point0(y_p0,survive_flag)
                best_group.append([p1,p0,b,log_reg,best_threshold,y_p0])
            if k==kmax:
                judge1=1
        best_group.sort(reverse=False,key=lambda element: element[0])
        best_finn=[]
        for single in best_group:
            if single[0]==best_group[0][0]:
                best_finn.append(single)
        best_finn.sort(reverse=False,key=lambda element: element[1])
        
        best_fin=[]
        for single in best_finn:
            # if single[1]==best_group[0][1]:#如果不对劲就再改回来
            if single[1]==best_finn[0][1]:
                best_fin.append(single)
        best_fin.sort(reverse=False,key=lambda element: len(element[2]))
        
        
        repeat.append(len(best_fin))
        house.append(best_fin[0])
        #把剩下的survive输出出来
        y_p0=best_fin[0][5]
        # print(y_p0)
        # print(survive_flag)
        if count_point(y_p0,survive_flag)==0:
            print('nomal end')
            break
        else:
            s=[]
            ss=[]
            for i in range(len(y_p0)):
                if y_p0[i]==0:
                    s.append(survive[i])
                    ss.append(survive_flag[i])
            survive=s[:]
            survive_flag=ss[:]

            print('next round survive number',len(survive_flag))
            print('next round 1 number',sum(survive_flag))
            print('next round 0 number',len(survive_flag)-sum(survive_flag))
            if sum(survive_flag)==len(survive_flag) or sum(survive_flag)==0 or m==mmax:
                print('end')
                break
            print('\n')
        
    # save_variable(house,'house.pickle')
    # [p1,p0,model,log_reg,best_threshold,y_p0]
    return house,repeat

def newsmall(house,X,Y,dna_name):
    choose_gene=[]
    choose_id=[]
    yy=[]
    for model in house:
        dna=[]
        for i in model[2]:
            dna.append(dna_name[i])
            # print('66666',c)
            if dna_name[i] not in choose_gene:
                choose_gene.append(dna_name[i])
            if i not in choose_id:
                choose_id.append(i)
        print('~~~~~~~~~~~')
        print('Gene Symbol:',dna)
        print('ID:',model[2])
        print('coef',model[3].coef_)
        print('intercept',model[3].intercept_)
        print('threshold',model[4])
        index=model[2]
        x=[]
        for i in range(len(X)):
            xx=[]
            for j in index:
                xx.append(X[i][j])
            x.append(xx)
        # [p1,p0,b,log_reg,best_threshold,y_p0]
        y_pred = model[3].predict_proba(x)[:, 1]
        y_p0= (y_pred  >= model[4]).astype(int)
        yy.append(y_p0)

    y=[]
    for i in range(len(yy[0])):#group1 和 group2
        yyy=[]
        for m in yy:
            yyy.append(m[i])
        y.append(int(max(yyy)))

    return count_point(y,Y),y,choose_gene,choose_id

# 直接使用index，而不再寻找
# [p1,p0,model,log_reg,best_threshold,y_p0]
def newVenn(house,X,Y,title='Venn3'):
    Set=[]
    SSet=[]
    for model in house[:3]:

        index=model[2]
        x=[]
        for i in range(len(X)):
            xx=[]
            for j in index:
                xx.append(X[i][j])
            x.append(xx)

        y_pred = model[3].predict_proba(x)[:, 1]
        y_p0= (y_pred  >= model[4]).astype(int)

        # print(Y)
        s=[]
        ss=[]
        for i in range(len(y_p0)):
            # print(y_p0[i],Y[i])
            if y_p0[i]==Y[i]:
                s.append(str(i))
                if y_p0[i]==1:
                    ss.append(str(i))
        Set.append(set(s))
        SSet.append(set(ss))
        # print(SSet)
        # print('next model\n')
    # 绘制三个集合的维恩图

    try:
        plt.figure(figsize=(8, 8))
        venn = venn3(Set, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$', r'$sm_'+str(3)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(24)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(24)
        # plt.title('Venn Diagram of Three Sets(with 0)')
        plt.savefig(title+'_01.png')
        # plt.show()
        plt.close()
    except:
        plt.figure(figsize=(8, 8))
        venn = venn2(Set, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(24)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(24)
        # plt.title('Venn Diagram of Three Sets(with 0)')
        plt.savefig(title+'_01.png')
        # plt.show()
        plt.close()

    # 绘制三个集合的维恩图
    try:
        plt.figure(figsize=(8, 8))
        # print(SSet)
        venn = venn3(SSet, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$', r'$sm_'+str(3)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(24)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(24)
        # plt.title('Venn Diagram of Three Sets(only 1)')
        plt.savefig(title+'_1.png')
        # plt.show()
        plt.close()
    except:
        plt.figure(figsize=(8, 8))
        # print(SSet)
        venn = venn2(SSet, (r'$sm_'+str(1)+'$', r'$sm_'+str(2)+'$'))
        # 调整字体大小
        for text in venn.set_labels:
            text.set_fontsize(24)
        for text in venn.subset_labels:
            if text:  # 检查文本对象是否存在
                text.set_fontsize(24)
        # plt.title('Venn Diagram of Three Sets(only 1)')
        plt.savefig(title+'_1.png')
        # plt.show()
        plt.close()





import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
# ========== 1. 特征选择方法（返回索引） ==========

def feature_selection_methods(X, y, k=50):
    """
    输入:
        X: (n, p) numpy数组 / list
        y: (n,) numpy数组 / list
        k: 选取的特征数量
    输出:
        一个字典 {方法名: 选中特征索引list}
    """
    X = np.array(X)
    y = np.array(y)

    # chi2 要求非负
    X_minmax = MinMaxScaler().fit_transform(X)

    results = {}

    # (1) 卡方
    chi2_selector = SelectKBest(chi2, k=min(k, X.shape[1])).fit(X_minmax, y)
    results["chi2"] = chi2_selector.get_support(indices=True).tolist()

    # (2) 方差分析 F-score
    f_selector = SelectKBest(f_classif, k=min(k, X.shape[1])).fit(X, y)
    results["anova"] = f_selector.get_support(indices=True).tolist()

    # (3) 互信息
    mi_selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1])).fit(X, y)
    results["mutual_info"] = mi_selector.get_support(indices=True).tolist()

    # (4) 不做筛选，保留所有
    results["all"] = list(range(X.shape[1]))
    print(results)
    return results


# ========== 2. 分类器集合 ==========

def classification_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "SVM": SVC(probability=True),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
    }


# ========== 3. 评估函数（交叉验证 + resubstitution） ==========

def evaluate_models(X, y, k=50, cv=5):
    """
    输入:
        X: (n, p)
        y: (n,)
        k: 选择的特征数量
        cv: 折数（默认5），如果想关掉交叉验证，可以单独跑resubstitution
    输出:
        结果字典 {特征选择方法: {分类器: (cv_acc, resub_acc)}}
    """
    X = np.array(X)
    y = np.array(y)

    results = {}
    fs_results = feature_selection_methods(X, y, k=k)
    models = classification_models()

    for fs_name, selected_idx in fs_results.items():
        X_sel = X[:, selected_idx]  # 取出被选中的列
        results[fs_name] = {}

        print(f"\n>>> 特征选择方法: {fs_name}, 特征数: {len(selected_idx)}")

        for model_name, model in models.items():
            # 交叉验证
            scores = cross_val_score(model, X_sel, y, cv=cv, scoring="accuracy")
            cv_acc = np.mean(scores)

            # resubstitution
            model.fit(X_sel, y)
            y_pred = model.predict(X_sel)
            resub_acc = accuracy_score(y, y_pred)

            results[fs_name][model_name] = (cv_acc, resub_acc)

            print(f"  {model_name:15s}: CV均值 = {cv_acc:.4f}, Resub = {resub_acc:.4f}")

    return results


#k折交叉验证，新初筛，更新mix方式版本
def newkfolds(X,Y,dna_name,key,gene,k=5,r=10):

    
    index1=0
    index0=0
    for i in range(len(Y)):
        if Y[i]==1:
            index1=i
            break
    for i in range(len(Y)):
        if Y[i]==0:
            index0=i
            break
    num1=index0
    num0=len(Y)-index0
    a1=knum(num1,k)
    a0=knum(num0,k)
    error=[]
    # print(index1,index0)
    # print(a1)
    # print(a0)
    y_p0=[]
    y_0=[]


    n=1
    for i in range(len(a1)):
        
        #测定index范围
        choose=[]#记录该人是否被选中训练
        for j in range(len(Y)):
            if j >=index1 and j<index1+a1[i]:
                choose.append(1)
            elif j >=index0 and j<index0+a0[i]:
                choose.append(1)
            else:
                choose.append(0)
        # print(choose)
        index1+=a1[i]
        index0+=a0[i]

        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        for j in range(len(choose)):
            if choose[j]==0:
                y_train.append(Y[j])
                x_train.append(X[j])
            elif choose[j]==1:
                y_test.append(Y[j])
                x_test.append(X[j])

        house,repeat=newbig(x_train,y_train,dna_name,r=r)
        b,y,c,d=newsmall(house,x_test,y_test,dna_name)
        four=ss(y_test,y)
        wen=[
            'True Positive Rate(sensitivity)',
            'True Negative Rate(specificity)',
            'False Positive Rate',
            'False Negative Rate',
        ]
        for i in range(len(wen)):
            print(wen[i],':',four[i])
        error.append(b)
        print('test',b)
        y_p0+=y
        y_0+=y_test
        b,y,c,d=newsmall(house,x_train,y_train,dna_name)
        print('self',c)
        print('repeat',repeat)
        print('include ID',d)
        print('include gene',c)
        print('gene count',len(c))
        print('error count',b)
        print('accurancy',1-b/len(X))
        

        
        
        print(str(n)+'!!!!!!!!!!!!!!!!!!!')

        n+=1
        print('\n')
    print('\nfinal')
    print(error)
    print(sum(error))
    print('accuracy',1-sum(error)/len(Y))
    try:
        print(ss(y_p0,y_0))
    except:
        print('TNR = TN /(TN + FP): division by zero')

def newkfolds2(X,Y,dna_name,key,gene,k=5,r=20):

    
    index1=0
    index0=0
    for i in range(len(Y)):
        if Y[i]==1:
            index1=i
            break
    for i in range(len(Y)):
        if Y[i]==0:
            index0=i
            break
    num1=index0
    num0=len(Y)-index0
    a1=knum(num1,k)
    a0=knum(num0,k)
    error=[]
    # print(index1,index0)
    # print(a1)
    # print(a0)
    y_p0=[]
    y_0=[]


    n=1
    for i in range(len(a1)):
        
        #测定index范围
        choose=[]#记录该人是否被选中训练
        for j in range(len(Y)):
            if j >=index1 and j<index1+a1[i]:
                choose.append(1)
            elif j >=index0 and j<index0+a0[i]:
                choose.append(1)
            else:
                choose.append(0)
        # print(choose)
        index1+=a1[i]
        index0+=a0[i]

        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        for j in range(len(choose)):
            if choose[j]==0:
                y_train.append(Y[j])
                x_train.append(X[j])
            elif choose[j]==1:
                y_test.append(Y[j])
                x_test.append(X[j])

        house,repeat=newbig2(x_train,y_train,dna_name,r=r)
        b,y,c,d=newsmall(house,x_test,y_test,dna_name)
        four=ss(y_test,y)
        wen=[
            'True Positive Rate(sensitivity)',
            'True Negative Rate(specificity)',
            'False Positive Rate',
            'False Negative Rate',
        ]
        for i in range(len(wen)):
            print(wen[i],':',four[i])
        error.append(b)
        print('test',b)
        y_p0+=y
        y_0+=y_test
        b,y,c,d=newsmall(house,x_train,y_train,dna_name)
        print('self',c)
        print('repeat',repeat)
        print('include ID',d)
        print('include gene',c)
        print('gene count',len(c))
        print('error count',b)
        print('accurancy',1-b/len(X))
        

        
        
        print(str(n)+'!!!!!!!!!!!!!!!!!!!')

        n+=1
        print('\n')
    print('\nfinal')
    print(error)
    print(sum(error))
    print('accuracy',1-sum(error)/len(Y))
    try:
        print(ss(y_p0,y_0))
    except:
        print('TNR = TN /(TN + FP): division by zero')


#老的初筛方法
def old(X,y):
    X = np.array(X)  # 转为numpy数组，形状 (n, p)
    y = np.array(y)
    n, p = X.shape
    scores = []
    # l=sum(y)
    for j in range(p):
        #01无所谓，只要分成了两类就行
        xx=[[i] for i in X[:, j]]
        log_reg=LR(xx,y)
        y_pred = log_reg.predict_proba(xx)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        j_scores = tpr - fpr
        best_threshold_index = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_index]
        y_p0= (y_pred  >= best_threshold).astype(int)
        # igr = information_gain_ratio(y_p0, y)
        p0=count_point(y_p0,y)
        scores.append([j, p0]) 
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return scores_sorted 


# ========== 新增：统计模型实际用到的特征数 ==========
def get_n_used_features(model, X):
    """返回模型实际使用的特征数"""
    if hasattr(model, "coef_"):  # 线性模型（逻辑回归、线性SVM等）
        return int(np.sum(np.abs(model.coef_) > 1e-6))
    elif hasattr(model, "feature_importances_"):  # 树模型、随机森林、XGBoost
        return int(np.sum(model.feature_importances_ > 0))
    else:  # 默认认为全用到了
        return X.shape[1]