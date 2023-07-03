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
class FeatureExtractor(nn.Module):
    """
    Embedding layer for encoding categorical variables.
    """

    def __init__(self, cate_fea_nuniqs, input_dim, emb_size):
        """
        Args:
            embedding_sizes (List[Tuple[int, int]]): List of (Unique categorical variables + 1, embedding dim)
        """
        super().__init__()
        self.input_dims = input_dim + len(cate_fea_nuniqs) * emb_size
        self.sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs
        ])
        # self.input_layer = nn.Linear(self.input_dims, extract_dim)

    def forward(self, X_sparse, X_dense):
        data_emb = [embedding_layer(X_sparse[:, i]) for i, embedding_layer in enumerate(self.sparse_emb)]
        data_cat = data_emb + [X_dense]
        out = torch.cat(data_cat, dim = 1)
        return out

class CtrNetwork(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, cate_fea_nuniqs, input_dim, emb_size, hid_dims = [256, 128, 64, 32], dropout=[0, 0, 0.2, 0.2], num_classes = 1):
        super().__init__()
        self.input_dims = input_dim + len(cate_fea_nuniqs) * emb_size
        self.all_dims = [self.input_dims] + hid_dims
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)

    def forward(self, X_emb):
        dnn_out = X_emb
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)
        out = self.sigmoid(dnn_out)
        return out


class CvrNetwork(nn.Module):
    """NN for CVR prediction"""

    def __init__(self, cate_fea_nuniqs, input_dim, emb_size, hid_dims = [256, 128, 64, 32], dropout=[0, 0, 0.2, 0.2], num_classes = 1):
        super().__init__()
        self.input_dims = input_dim + len(cate_fea_nuniqs) * emb_size
        self.all_dims = [self.input_dims] + hid_dims
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)

    def forward(self, X_emb):
        dnn_out = X_emb
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)

        dnn_out = self.dnn_linear(dnn_out)
        out = self.sigmoid(dnn_out)
        return out


class ESMM(nn.Module):
    """ESMM"""

    def __init__(self, cate_fea_nuniqs, input_dim, emb_size, hid_dims = [256, 128, 64, 32], dropout=[0, 0, 0.2, 0.2], num_classes = 1):
        super().__init__()
        self.feature_extractor = FeatureExtractor(cate_fea_nuniqs, input_dim, emb_size)

        self.ctr_network = CtrNetwork(cate_fea_nuniqs, input_dim, emb_size, hid_dims, dropout, num_classes)
        self.cvr_network = CvrNetwork(cate_fea_nuniqs, input_dim, emb_size, hid_dims, dropout, num_classes)

    def forward(self, X_sparse, X_dense):
        # embedding
        out = self.feature_extractor(X_sparse, X_dense)
        # Predict pCTR
        p_ctr = self.ctr_network(out)
        # Predict pCVR
        p_cvr = self.cvr_network(out)
        # Predict pCTCVR
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return p_ctr, p_ctcvr

model = ESMM(cate_fea_nuniqs= cate_fea_nuniqs, input_dim = len(dense_feature) , emb_size = 12, hid_dims = [64, 64, 24], dropout=[0, 0, 0.1], num_classes = 1)
loss_fcn = nn.BCELoss()  # Loss函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_and_eval(model, train_loader, valid_loader, 100, device)
