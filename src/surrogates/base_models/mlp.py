from tools.stats import get_correlation

import torch.nn as nn
import numpy as np
import torch
import copy


class Net(nn.Module):
    def __init__(self, input_dim, n_layers=3, n_hidden=450, n_output=1, drop=0.05):
        super(Net, self).__init__()

        self.stem = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.ReLU())

        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(n_hidden, n_hidden))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        self.drop = nn.Dropout(p=drop)
        self.regressor = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.stem(x)
        x = self.hidden(x)
        x = self.drop(x)
        x = self.regressor(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) is nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


class MLP:
    """Multi Layer Perceptron"""

    def __init__(self, **kwargs):
        self.model = Net(**kwargs)
        self.name = "mlp"

    def fit(self, **kwargs):
        self.model = train(self.model, **kwargs)

    def predict(self, test_data, device="cpu"):
        return predict(self.model, test_data, device=device)


def train(
    net,
    x,
    y,
    lr=1e-3,
    epochs=2000,
    device="cuda",
):
    # Prepare the inputs and targets
    _inputs = torch.from_numpy(x).float()
    _target = torch.from_numpy(y).unsqueeze(1).float()

    # Standardize the inputs
    _inputs = (_inputs - _inputs.mean()) / _inputs.std()

    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(epochs),
        eta_min=0,
    )

    best_tau = float("-inf")
    for _ in range(epochs):
        train_one_epoch(
            net,
            _inputs,
            _target,
            criterion,
            optimizer,
            device,
        )
        scheduler.step()

        _, _, tau = validate(
            net,
            _inputs,
            _target,
            device,
        )

        if tau > best_tau:
            best_tau = tau
            best_net = copy.deepcopy(net)

    return best_net.to("cpu")


def train_one_epoch(net, data, target, criterion, optimizer, device):
    net.train()
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)
    pred = net(data)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(net, data, target, device):
    net.eval()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        pred = net(data)
        pred, target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()

        rmse, rho, tau = get_correlation(pred, target)

    return rmse, rho, tau


def predict(net, query, device):
    if query.ndim < 2:
        data = torch.zeros(1, query.shape[0])
        data[0, :] = torch.from_numpy(query).float()
    else:
        data = torch.from_numpy(query).float()

    net = net.to(device)
    net.eval()
    with torch.no_grad():
        data = data.to(device)
        pred = net(data)

    return pred.cpu().detach().numpy()
