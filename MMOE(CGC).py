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

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import torch.utils.data as Data
import numpy as np 
import time
from tqdm import tqdm
# embedding
class FeatureExtractor(nn.Module):
    """
    Embedding layer for encoding categorical variables.
    """

    def __init__(self, cate_fea_nuniqs, emb_size):
        super().__init__()
        self.sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs
        ])

    def forward(self, X_sparse, X_dense):
        data_emb = [embedding_layer(X_sparse[:, i]) for i, embedding_layer in enumerate(self.sparse_emb)]
        data_cat = data_emb + [X_dense]
        out = torch.cat(data_cat, dim = 1)
        return out

class Expert(nn.Module):
    def __init__(self, input_size, experts_hid, dropout):
        super(Expert, self).__init__()
        self.experts_hid = experts_hid
        self.fc = nn.Linear(input_size, self.experts_hid[0])
        self.relu = nn.ReLU()

        for i in range(1, len(self.experts_hid)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.experts_hid[i-1], self.experts_hid[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.experts_hid[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))

    def forward(self, x):
        exp_out = self.relu(self.fc(x))
        for i in range(1, len(self.experts_hid)):
            exp_out = getattr(self, 'linear_' + str(i))(exp_out)
            exp_out = getattr(self, 'batchNorm_' + str(i))(exp_out)
            exp_out = getattr(self, 'activation_' + str(i))(exp_out)
            exp_out = getattr(self, 'dropout_' + str(i))(exp_out)
        return exp_out

class Tower(nn.Module):
    def __init__(self, input_size, tower_hid, dropout):
        super(Tower, self).__init__()
        self.tower_hid = tower_hid
        self.fc1 = nn.Linear(input_size, self.tower_hid[0])
        self.relu = nn.ReLU()

        for i in range(1, len(self.tower_hid)):
            setattr(self, 'tower_linear_'+str(i), nn.Linear(self.tower_hid[i-1], self.tower_hid[i]))
            setattr(self, 'tower_batchNorm_' + str(i), nn.BatchNorm1d(self.tower_hid[i]))
            setattr(self, 'tower_activation_' + str(i), nn.ReLU())
            setattr(self, 'tower_dropout_'+str(i), nn.Dropout(dropout))
        self.fc2 = nn.Linear(self.tower_hid[-1], 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        tower_out = self.fc1(x)
        tower_out = self.relu(tower_out)

        for i in range(1, len(self.tower_hid)):
            tower_out = getattr(self, 'tower_linear_' + str(i))(tower_out)
            tower_out = getattr(self, 'tower_batchNorm_' + str(i))(tower_out)
            tower_out = getattr(self, 'tower_activation_' + str(i))(tower_out)
            tower_out = getattr(self, 'tower_dropout_' + str(i))(tower_out)
        
        out = self.fc2(tower_out)
        out = self.sigmoid(out)
        return out
    
class CGC(nn.Module):
    def __init__(self, 
                input_size,
                cate_fea_nuniqs,
                emb_size, 
                num_specific_experts, 
                num_shared_experts, 
                experts_hid = [256, 128, 64, 32], 
                experts_dropout = [0, 0, 0.2, 0.2], 
                towers_hid = [16, 8], 
                towers_dropout = 0.2,
                num_tasks = 2
                ):
        super().__init__()
        self.input_size = input_size + len(cate_fea_nuniqs) * emb_size
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.experts_hid = experts_hid
        self.experts_dropout = experts_dropout
        self.towers_hid = towers_hid
        self.towers_dropout = towers_dropout
        self.feature_extractor = FeatureExtractor(cate_fea_nuniqs, emb_size)
        self.experts_shared = nn.ModuleList([Expert(self.input_size, self.experts_hid, self.experts_dropout) for i in range(self.num_shared_experts)])
        self.experts_task1 = nn.ModuleList([Expert(self.input_size, self.experts_hid, self.experts_dropout) for i in range(self.num_specific_experts)])
        self.experts_task2 = nn.ModuleList([Expert(self.input_size, self.experts_hid, self.experts_dropout) for i in range(self.num_specific_experts)])
        self.soft = nn.Softmax(dim=1)
        self.dnn1 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                 nn.Softmax(dim=1))
        self.dnn2 = nn.Sequential(nn.Linear(self.input_size, self.num_specific_experts + self.num_shared_experts),
                                  nn.Softmax(dim=1))
        self.tower1 = Tower(self.experts_hid[-1], self.towers_hid, self.towers_dropout)
        self.tower2 = Tower(self.experts_hid[-1], self.towers_hid, self.towers_dropout)


    def forward(self, X_sparse, X_dense):
        x = self.feature_extractor(X_sparse, X_dense)
        experts_shared_o = [e(x) for e in self.experts_shared]
        experts_shared_o = torch.stack(experts_shared_o)
        experts_task1_o = [e(x) for e in self.experts_task1]
        experts_task1_o = torch.stack(experts_task1_o)
        experts_task2_o = [e(x) for e in self.experts_task2]
        experts_task2_o = torch.stack(experts_task2_o)

        # gate1
        selected1 = self.dnn1(x)
        gate_expert_output1 = torch.cat((experts_task1_o, experts_shared_o), dim=0)
        gate1_out = torch.einsum('abc, ba -> bc', gate_expert_output1, selected1)
        final_output1 = self.tower1(gate1_out)

        # gate2
        selected2 = self.dnn2(x)
        gate_expert_output2 = torch.cat((experts_task2_o, experts_shared_o), dim=0)
        gate2_out = torch.einsum('abc, ba -> bc', gate_expert_output2, selected2)
        final_output2 = self.tower2(gate2_out)

        return final_output1, final_output2
import time
from tqdm import tqdm
def train_and_eval(model, train_loader, valid_loader, epochs, device):
    best_auc1, best_auc2 = 0.0, 0.0
    for i in range(1,1+epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(i))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, nume_fea, label1, label2 = x[0], x[1], x[2],x[3]
#             cate_fea, nume_fea, label = cate_fea.to(device), nume_fea.to(device), label.float().to(device)
            pred1, pred2 = model(cate_fea, nume_fea)#.view(-1)
            loss1 = loss_fcn(pred1, label1)
            loss2 = loss_fcn(pred2, label2)
            loss = loss1 + loss2
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
            torch.save(model.state_dict(), "model/best_4.pth")
        torch.save(model, "model/data/model_4_" + str(i) + ".pt")
        write_log('Current CTR AUC: %.6f, Best AUC: %.6f\n, Current CTRVR AUC: %.6f, Best AUC: %.6f\n' % (cur_auc1, best_auc1, cur_auc2, best_auc2))


from sklearn.metrics import roc_auc_score,roc_curve
from matplotlib import pyplot as plt
model = CGC(input_size = len(dense_feature), cate_fea_nuniqs = cate_fea_nuniqs, emb_size = 12, num_specific_experts = 1, num_shared_experts = 2, experts_hid = [64, 64, 32], experts_dropout = [0, 0, 0.1])
loss_fcn = nn.BCELoss()  # Loss函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_and_eval(model, train_loader, valid_loader, 100, device)
