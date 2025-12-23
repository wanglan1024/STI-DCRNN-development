import torch.utils.data as Data
import torch
from torch import nn, optim
from torch_geometric_temporal import DCRNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


class Config:
    def __init__(self):
        self.tag_data_path = './data/Chla.csv'
        self.data_path = './data/STfill_Chla.csv'
        self.time_delay = 6
        self.step = 1 #预测步长
        self.scale = MinMaxScaler()
        self.train_test_split_rate = 0.8
        #self.input_dim = ； # 输入特征乘以回看窗口
        self.hidden_dim = 64
        self.learning_rate = 0.0001
        self.epoches = 200
        self.batch_size = 32
        self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")


class my_data(Data.Dataset):
    def __init__(self, or_data, lags, step, tag_data=None):
        super(my_data, self).__init__()
        self.data = or_data
        self.tag_data = tag_data
        self.lags = lags
        self.step = step

    def __getitem__(self, item):
        data_x = self.data[item:item + self.lags, :]
        data_y = self.data[item + self.lags + self.step - 1, :]
        if self.tag_data is None:
            return data_x, data_y, False
        else:
            tag = self.tag_data[item + self.lags + self.step - 1, :]
            return data_x, data_y, tag

    def __len__(self):
        return self.data.shape[0] - self.lags - self.step + 1


class dcrnn(nn.Module):
    def __init__(self, node_features, hidden_dim, K=2):
        super(dcrnn, self).__init__()

        self.recurrent = DCRNN(node_features, hidden_dim, K)
        self.activate = nn.ReLU()
        #self.dropout = nn.Dropout(p=0.05)
        self.linear = nn.Linear(hidden_dim, 1)


    def forward(self, x, edge_index, edge_weight):
        out = self.recurrent(x, edge_index, edge_weight)
        out = self.activate(out)
      #  out = self.dropout(out)  # 在需要添加Dropout的地方使用Dropout层
        out = self.linear(out)
        return out


def my_dataset(args):
    or_data = pd.read_csv(args.data_path, index_col=None, header=None, encoding='gbk').values.astype(np.float32)
    or_tag = pd.read_csv(args.tag_data_path, index_col=None, header=0, encoding='gbk').values
    train_test_split = round(or_data.shape[0] * args.train_test_split_rate)
    or_train = or_data[:train_test_split, :]
    or_test = or_data[train_test_split:, :]

    args.scale.fit(or_train)
    scale_train_data = args.scale.transform(or_train)
    scale_test_data = args.scale.transform(or_test)

    start_tag = args.time_delay + args.step - 1
    scale_test_data = np.vstack((scale_train_data[-start_tag:, :], scale_test_data))

    tag_data = or_tag[:or_train.shape[0], :]
    tag_test_data = np.vstack((tag_data[-start_tag:, :], or_tag[or_train.shape[0]:, :]))

    train_dataloader = my_data(scale_train_data, args.time_delay, args.step)
    test_dataloader = my_data(scale_test_data, args.time_delay, args.step, tag_test_data)

    train_dataset = Data.DataLoader(train_dataloader, collate_fn=collate_fn)
    test_dataset = Data.DataLoader(test_dataloader, collate_fn=collate_fn)

    edges = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 4], [4, 1], [2, 3], [3, 2], [3, 4], [4, 3],
             [2, 5], [5, 2], [3, 5], [5, 3], [5, 6], [6, 5], [5, 7], [7, 5], [6, 7], [7, 6]]
    edges_numpy = np.array(edges, dtype=np.int64).T
    edges_tensor = torch.tensor(edges_numpy)

    #edge_weight = torch.ones(len(edges), dtype=torch.float32)
    edge_weight = torch.tensor([0.1540, 0.1160, 0.1540, 0.1669, 0.1669,
                                0.1917, 0.1917, 0.2932, 0.2002, 0.1160,
                                0.2932, 0.3250, 0.2002, 0.3250, 0.1742,
                                0.0833, 0.1742, 0.1339, 0.0833, 0.1339], dtype=torch.float32)
    return train_dataset, test_dataset, edges_tensor, edge_weight


def collate_fn(batch):
    data_x, data_y, tag = zip(*batch)
    data_x = torch.tensor(np.array(data_x)).squeeze(0).transpose(0, 1)
    data_y = torch.tensor(np.array(data_y)).squeeze(0)
    tag = ~torch.tensor(np.array(tag)).squeeze(0).isnan()
    return data_x, data_y, tag


def model_train(model, train_data_set, test_data_set, edges, weights, criterion, optimizer, args):
    loss_train_epoch = []
    loss_test_epoch = []
    plt.ion()
    for epoch in range(args.epoches):
        loss_train_iteration = 0.0
        loss_test_iteration = 0.0
        train_sample_num = 0
        test_sample_num = 0
        model.train() #训练
        loss = 0.
        for iteration, (train_x, train_y, _) in enumerate(train_data_set):
            train_x, train_y = train_x.to(args.device), train_y.to(args.device)
            y_hat = model.forward(train_x, edges, weights)
            loss += criterion(y_hat.squeeze(), train_y)
            loss_train_iteration += loss.item()
            if (iteration + 1) % args.batch_size == 0:
                loss.backward()
                optimizer.step()
                loss = 0.
            train_sample_num += 1

        for _, (test_x, test_y, tag) in enumerate(test_data_set):
            test_x = test_x.to(args.device)
            y_hat = model.forward(test_x, edges, weights)
            loss_mse = criterion(y_hat.cpu().squeeze().masked_select(tag), test_y.masked_select(tag)).item()
            loss_test_iteration += loss_mse
            test_sample_num += 1

        loss_train_epoch.append(loss_train_iteration / train_sample_num)
        loss_test_epoch.append(loss_test_iteration / test_sample_num)

        print('Epoch：{0}/{1}'.format(epoch + 1, args.epoches))
        print('Train MSE：{0:.4f}，Test MSE：{1:.4f}'.format(loss_train_epoch[-1], loss_test_epoch[-1]))
        print('-' * 50)
        draw_train(loss_train_epoch, loss_test_epoch)
    print('Finish training')
    return model


def model_val(model, data_set, edges, weights, args):
    criter_mse = nn.MSELoss()
    criter_mae = nn.L1Loss()
    pre_scale_y = []
    true_scale_y = []
    all_tag = []
    model.eval()
    for _, (train_x, train_y, tag) in enumerate(data_set):
        train_x, train_y = train_x.to(args.device), train_y.to(args.device)
        y_hat = model.forward(train_x, edges, weights)
        pre_scale_y.append(list(np.array(y_hat.cpu().squeeze().data)))
        true_scale_y.append(list(np.array(train_y.cpu().data)))
        if tag.shape != torch.Size([]):
            all_tag.append(list(np.array(tag.data)))
    all_tag = torch.tensor(np.array(all_tag))
    pre_y = args.scale.inverse_transform(pre_scale_y)
    true_y = args.scale.inverse_transform(true_scale_y)

    if all_tag.shape == torch.Size([0]):
        loss_mse = criter_mse(torch.tensor(true_y, dtype=torch.float32),
                              torch.tensor(pre_y, dtype=torch.float32))
        loss_mae = criter_mae(torch.tensor(true_y, dtype=torch.float32),
                              torch.tensor(pre_y, dtype=torch.float32))
        separate_mse = [criter_mse(torch.tensor(true_y[:, i], dtype=torch.float32),
                                   torch.tensor(pre_y[:, i], dtype=torch.float32)) for i in range(true_y.shape[1])]
        separate_mae = [criter_mae(torch.tensor(true_y[:, i], dtype=torch.float32),
                                   torch.tensor(pre_y[:, i], dtype=torch.float32)) for i in range(true_y.shape[1])]

        mean_r2 = [r2_score(true_y[:, i], pre_y[:, i]) for i in range(true_y.shape[1])]
        all_r2 = r2_score(true_y.reshape(-1),pre_y.reshape(-1))


    else:
        loss_mse = criter_mse(torch.tensor(true_y, dtype=torch.float32).masked_select(all_tag),
                              torch.tensor(pre_y, dtype=torch.float32).masked_select(all_tag))
        loss_mae = criter_mae(torch.tensor(true_y, dtype=torch.float32).masked_select(all_tag),
                              torch.tensor(pre_y, dtype=torch.float32).masked_select(all_tag))
        r2 = r2_score(torch.tensor(true_y).masked_select(all_tag),torch.tensor(pre_y).masked_select(all_tag))
        separate_mse = [criter_mse(torch.tensor(true_y[:, i], dtype=torch.float32).masked_select(all_tag[:, i]),
                                   torch.tensor(pre_y[:, i], dtype=torch.float32).masked_select(all_tag[:, i])) for i in
                        range(true_y.shape[1])]
        separate_mae = [criter_mae(torch.tensor(true_y[:, i], dtype=torch.float32).masked_select(all_tag[:, i]),
                                   torch.tensor(pre_y[:, i], dtype=torch.float32).masked_select(all_tag[:, i])) for i in
                        range(true_y.shape[1])]
        mean_r2 = [r2_score(torch.tensor(true_y[:, i]).masked_select(all_tag[:, i]),
                           torch.tensor(pre_y[:, i]).masked_select(all_tag[:, i])) for i in range(true_y.shape[1])]
        all_r2 = r2_score(torch.tensor(true_y.reshape(-1)).masked_select(all_tag.reshape(-1)),
                           torch.tensor(pre_y.reshape(-1)).masked_select(all_tag.reshape(-1)))
    print('Total RMSE {:.4f}'.format(loss_mse.item() ** 0.5))
    print('Total MAE {:.4f}'.format(loss_mae.item()))
    print('Total R2 {:.4f}'.format(all_r2))
    print('Mean RMSE', [round(i.item() ** 0.5, 4) for i in separate_mse])
    print('Mean MAE', [round(i.item(), 4) for i in separate_mae])
    print('mean_r2', np.around(mean_r2, 4))
    for i in range(true_y.shape[1]):
        draw_result(mean_r2, true_y[:, i], pre_y[:, i], mean_r2[i])


def draw_train(a, b):
    plt.figure(1, figsize=(6, 6))
    plt.clf()
    plt.plot(range(len(a)), a, 'ro-', label='Train loss')
    plt.plot(range(len(b)), b, 'bs-', label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.pause(0.005)
    plt.show()


def draw_result(all_r2, x, y, r2):
    plt.figure(2, figsize=(6, 6))
    plt.clf()
    plt.bar(range(len(all_r2)), all_r2)
    plt.figure(3, figsize=(6, 6))
    plt.clf()
    plt.plot(x, 'b-', label='origin data')
    plt.plot(y, 'r-', label='predict data')
    plt.xlabel('sample')
    plt.ylabel('value')
    plt.title('r2={0:.4f}'.format(r2))
    plt.legend()
    plt.show()


def main(pattern='test'):
    args = Config()
    train_dataset, test_dataset, edges_tensor, edge_weight = my_dataset(args)
    edges_tensor, edge_weight = edges_tensor.to(args.device), edge_weight.to(args.device)

    model = dcrnn(args.time_delay, args.hidden_dim).to(args.device)
    total_param = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %d' % (total_param))

    if pattern == 'train':
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        model = model_train(model, train_dataset, test_dataset, edges_tensor, edge_weight, criterion, optimizer, args)
        torch.save(model.state_dict(), 'DCRNN_Pre_01.pth')
    else:
        model.load_state_dict(torch.load('DCRNN_Pre_01.pth'))
        model_val(model, train_dataset, edges_tensor, edge_weight, args)
        model_val(model, test_dataset, edges_tensor, edge_weight, args)


if __name__ == '__main__':
    main('train')
    print('程序运行结束')
    plt.pause(0)
