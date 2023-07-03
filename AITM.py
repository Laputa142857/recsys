# ------load data----
import pickle
with open('model/3-browser-data.pkl', 'rb') as f:
    data1, data2, data3 = pickle.load(f)
data1 = data1[head_import]
data2 = data2[head_import]
data3 = data3[head_import]
tmp = data1.append(data2)
data = tmp.append(data3)

# -------- preprocess ----------
data[dense_feature] = data[dense_feature].fillna(0, )
data[dense_feature] = data[dense_feature].values.astype(float)
data[sparse_feature] = data[sparse_feature].astype(str)
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_feature] = mms.fit_transform(data[dense_feature])
store = dict()
for feat in sparse_feature:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
    store[feat] = lbe
cate_fea_nuniqs = [data[f].nunique() for f in sparse_feature]

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import torch.utils.data as Data
import numpy as np 

num1,fea1 = data1.shape
num2,fea2 = data2.shape
num3,fea3 = data3.shape
print(num1,fea1, num2, num3)
y1 = np.zeros((num1 + num2 + num3,1))
for i in range(num1 + num2):
    y1[i] = 1
y2 = np.zeros((num1 + num2 + num3,1))
for i in range(num1):
    y2[i] = 1
x = data

output = open('model/ms4.pkl', 'wb')
pickle.dump([mms, store], output)
output.close()
for feat in sparse_feature:
    lbe = store[feat]
    data[feat] = lbe.fit_transform(data[feat])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train1, y_test1 , y_train2, y_test2 = train_test_split(x, y1, y2, test_size=0.2, random_state=369)
X_val, X_test, y_val1, y_test1, y_val2, y_test2 = train_test_split(X_test, y_test1, y_test2, test_size=0.3, random_state=258)

# -------data loader-------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 30
train_dataset = Data.TensorDataset(torch.LongTensor(X_train[sparse_feature].values), 
                                   torch.FloatTensor(X_train[dense_feature].values),
                                   torch.FloatTensor(y_train1),
                                  torch.FloatTensor(y_train2))

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = Data.TensorDataset(torch.LongTensor(X_val[sparse_feature].values), 
                                   torch.FloatTensor(X_val[dense_feature].values),
                                   torch.FloatTensor(y_val1),
                                  torch.FloatTensor(y_val2))
valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = Data.TensorDataset(torch.LongTensor(X_test[sparse_feature].values), 
                                   torch.FloatTensor(X_test[dense_feature].values),
                                   torch.FloatTensor(y_test1),
                                 torch.FloatTensor(y_test2))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ----------log -------
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def write_log(w):
    file_name = 'model/data/' + datetime.date.today().strftime('%m%d')+"_{}.log".format("deepfm")
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    # print(info)
    with open(file_name, 'a') as f: 
        f.write(info + '\n')

# -------model------------

constraint_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.cpu().item()
            #             if (idx+1) % 50 == 0 or (idx + 1) == len(train_loader):
            #                 write_log("Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
            #                           _+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start_time))
        scheduler.step()
        """推断部分"""
        model.eval()
        with torch.no_grad():
            valid_labels1, valid_preds1 = [], []
            valid_labels2, valid_preds2 = [], []
            for idx, x in tqdm(enumerate(valid_loader)):
                cate_fea, nume_fea, label1, label2 =  x[0], x[1], x[2], x[3]
#                 cate_fea, nume_fea = cate_fea.to(device), nume_fea.to(device)
                pred1, pred2 = model(cate_fea, nume_fea)#.reshape(-1).data.numpy().tolist()
                valid_preds1.extend(pred1)
                valid_labels1.extend(label1.numpy().tolist())
                valid_preds2.extend(pred2)
                valid_labels2.extend(label2.numpy().tolist())
        cur_auc1 = roc_auc_score(valid_labels1, valid_preds1)
        cur_auc2 = roc_auc_score(valid_labels2, valid_preds2)
        if cur_auc1 > best_auc1:
            best_auc1 = cur_auc1
#             torch.save(model.state_dict(), "esmm_best.pth")
        if cur_auc2 > best_auc2:
            best_auc2 = cur_auc2
            torch.save(model.state_dict(), "model/best_666.pth")
        torch.save(model, "model/data/model_66_" + str(i) + ".pt")
        write_log('Current CTR AUC: %.6f, Best AUC: %.6f\n, Current CTRVR AUC: %.6f, Best AUC: %.6f\n' % (cur_auc1, best_auc1, cur_auc2, best_auc2))
model = AITM(cate_fea_nuniqs, input_dim = len(dense_feature), emb_size = 12, hid_dims = [128, 64, 24], dropout=[0, 0, 0.2], att_dim = 24)
loss_fcn = nn.BCELoss()  # Loss函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_and_eval(model, train_loader, valid_loader, 100, device)
