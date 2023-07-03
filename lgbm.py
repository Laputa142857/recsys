# ----------------oad data from hql or sql--------
data = data.values.astype(float)
num1,fea1 = data1.shape
num2,fea2 = data2.shape
print(len(data), len(data[0]), num1,num2)

# ---------------label and dataset ---------------------------------
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import torch.utils.data as Data
import numpy as np 
y = np.zeros((num1 + num2,1))
for i in range(num1):
    y[i] = 1
x = data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y.ravel() , test_size=0.3, random_state=369)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.3, random_state=258)
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)


# -----------model parameter------------------
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss','auc'],
#     'num_leaves':31,
#     'max_depth':5,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'num_threads':8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'lambda_l2':5,
    'reg_alpha':1,
    'random_state':42,
    'min_gain_to_split':0.2,
    'n_estimators':800,
    'subsample':0.8
}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1500,
                valid_sets=[lgb_train, lgb_eval],
                early_stopping_rounds=50,
                verbose_eval = 10,
               )
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score


# --------------------eval
#lgb_predict 要转换为label
auc = roc_auc_score(y_test, y_pred)
y_pred=np.where(y_pred > 0.5, 1, 0)
print(f"predict test AUC: ", auc)
print("acc:",accuracy_score(y_test, y_pred))
print("recall_score:",recall_score(y_test,y_pred))
import_pd=pd.DataFrame(zip(head_import,gbm.feature_importance()),columns=['feature','importance_socre'])
pd.set_option("display.max_columns", 2000)
pd.set_option("display.max_rows", 2000)
# import_pd.sort_values(by=['importance_socre'],ascending=False)
import_pd.sort_values(by=['importance_socre'],ascending=False).head(100)
gbm.save_model('model/20230506.txt', num_iteration=gbm.best_iteration)

# ------ result plot-------

from sklearn.metrics import roc_auc_score,roc_curve
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
t1,t2 = y_test, y_pred
print('AUC : ', roc_auc_score(t1,t2))
zipped = zip(t1,t2)
rank1 = sorted(zipped, key=lambda x: -x[1])
rank1 = torch.tensor(rank1)
l = len(rank1)
out = []
s1,s2 = 0,0
for i in range(9):
    a = rank1[l*i//10:l*(i+1)//10,0]
    s1 += sum(a)
    s2 += sum(t1)/10
    out.append(s1/s2)
out.append(1)
from matplotlib import pyplot as plt 
x = np.arange(1,11) 
plt.title('Lift (divided by Percentage)')
plt.plot(['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'], out)
plt.ylabel('Lift')
plt.xlabel('Top')
plt.show()
out = []
s1,s2 = 0,0
for i in range(10):
    a = rank1[100*i:100*(i+1),0]
    s1 += sum(a)
    s2 += 100
    out.append(s1/s2)
plt.title('Precentage of positive samples')
plt.plot(['first 100','200','300','400','500','600','700','800','900','1000'], out)
plt.ylabel('Per')
plt.xlabel('Num')
plt.show()
out = []
c1,c2 = [],[]
l1,l2 = 0,0
eps,index,i = 0.9,0,0
while eps>0 and i < l:
    if rank1[i,1]>eps:
        i+=1
    else:
        eps -= 0.1
        out.append(sum(rank1[:i,0])*len(t1)/(i * sum(t1)))
        c1.append(sum(rank1[:i,0]) - l1)
        l1 = sum(rank1[:i,0])
        c2.append(i- sum(rank1[:i,0]) -l2)
        l2 = i- sum(rank1[:i,0])
out.append(1)
c1.append(sum(rank1[:i,0]) - l1)
c2.append(i- sum(rank1[:i,0]) -l2)
plt.title('Lift (divided by score)')
plt.plot(['>0.9','>0.8','>0.7','>0.6','>0.5','>0.4','>0.3','>0.2','>0.1','>0'], out)
plt.ylabel('Lift')
plt.xlabel('Score')
plt.show()
plt.title('Data Distribution')
plt.plot(['>0.9','>0.8','>0.7','>0.6','>0.5','>0.4','>0.3','>0.2','>0.1','>0'], c2, label = 'Negative Data')
plt.plot(['>0.9','>0.8','>0.7','>0.6','>0.5','>0.4','>0.3','>0.2','>0.1','>0'], c1, label = 'Positive Data')
plt.legend()
plt.ylabel('Num')
plt.xlabel('Score')
plt.show()

fpr,tpr,thresholds=roc_curve(t1,t2)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % 0.81)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

for i in range(len(fpr)):
    if(i==0):
        KS_max=tpr[i]-fpr[i]
        best_thr=thresholds[i]
    elif (tpr[i]-fpr[i]>KS_max):
        KS_max = tpr[i] - fpr[i]
        best_thr = thresholds[i]
x = np.arange(1,len(fpr)+1)
plt.plot(x,fpr, label = 'KS = %0.2f' % KS_max)
plt.legend(loc = 'lower right')
plt.plot(x,tpr)
plt.title('KS Figure')
plt.show()

