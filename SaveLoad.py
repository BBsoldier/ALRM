#将变量保存下来
import pickle
#保存变量函数：
def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    # return filename
#读取变量函数：
def load_variable(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r
'''
保存变量和读取变量操作。

保存变量：将变量results保存在results.txt文件中。

filename = save_variable(results, 'results.txt')

读取变量：从results.txt中读取变量内容给results 。

results = load_variavle('results.txt')
'''