# ------load data----
import pickle
with open('model/2-browser-data.pkl', 'rb') as f:
    data1, data2 = pickle.load(f)
data1 = data1[head_import]
data2 = data2[head_import]
data = data1.append(data2)

for i in range(num1):
    y[i] = 1
x = data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=369)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.3, random_state=258)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 30

train_dataset = Data.TensorDataset(torch.LongTensor(X_train[sparse_feature].values), 
                                   torch.FloatTensor(X_train[dense_feature].values),
                                   torch.FloatTensor(y_train))

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = Data.TensorDataset(torch.LongTensor(X_val[sparse_feature].values), 
                                   torch.FloatTensor(X_val[dense_feature].values),
                                   torch.FloatTensor(y_val))
valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = Data.TensorDataset(torch.LongTensor(X_test[sparse_feature].values), 
                                   torch.FloatTensor(X_test[dense_feature].values),
                                   torch.FloatTensor(y_test))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 打印模型参数
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# 定义日志（data文件夹下，同级目录新建一个data文件夹）
def write_log(w):
    file_name = 'model/data/' + datetime.date.today().strftime('%m%d')+"_{}.log".format("deepfm")
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f: 
        f.write(info + '\n')


class DeepFM(nn.Module):
    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, emb_size=12, 
                 hid_dims=[256, 128, 64, 32], num_classes=1, dropout=[0, 0, 0.2, 0.2]): 
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况 
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size
        
        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示
        
        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示
        
        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN 
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))
        # for output 
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_sparse, X_dense=None):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """
        
        """FM 一阶部分"""
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs, 1]
        
        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense) 
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]
        
        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)
        
        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed    # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2 
        sub = square_sum_embed - sum_square_embed  
        sub = sub * 0.5   # [bs, emb_size]
        
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # [bs, 1]
        
        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)   # [bs, n * emb_size]
        
        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))   # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out   # [bs, n * emb_size]
        
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        
        dnn_out = self.dnn_linear(dnn_out)   # [bs, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out   # [bs, 1]
        out = self.sigmoid(out)
        return out

import time
from tqdm import tqdm
def train_and_eval(model, train_loader, valid_loader, epochs, device):
    best_auc = 0.0
    for i in range(1,1+epochs):
        """训练部分"""
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        write_log('Epoch: {}'.format(i))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            cate_fea, nume_fea, label = x[0], x[1], x[2]
#             cate_fea, nume_fea, label = cate_fea.to(device), nume_fea.to(device), label.float().to(device)
            pred = model(cate_fea, nume_fea)#.view(-1)
            loss = loss_fcn(pred, label)
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
            valid_labels, valid_preds = [], []
            for idx, x in tqdm(enumerate(valid_loader)):
                cate_fea, nume_fea, label = x[0], x[1], x[2]
#                 cate_fea, nume_fea = cate_fea.to(device), nume_fea.to(device)
                pred = model(cate_fea, nume_fea).reshape(-1).data.numpy().tolist()
                valid_preds.extend(pred)
                valid_labels.extend(label.numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), "model/deepfm_best.pth")
        torch.save(model, "model/data/model_LR_" + str(i) + ".pt")
        write_log('Current AUC: %.6f, Best AUC: %.6f\n' % (cur_auc, best_auc))
from sklearn.metrics import roc_auc_score,roc_curve
from matplotlib import pyplot as plt 
number_feature = [i for i in data.columns if i not in sparse_feature]


model = DeepFM(cate_fea_nuniqs, nume_fea_size=len(dense_feature),emb_size=12, 
                 hid_dims=[128, 128, 64, 32], num_classes=1, dropout=[0, 0, 0.1, 0.1])
# print(get_parameter_number(model))
loss_fcn = nn.BCELoss()  # Loss函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_and_eval(model, train_loader, valid_loader, 100, device)
