# tests/test_client_server.py
import torch
import torch.nn as nn
import numpy as np
from src.fl.baseclient import BenignClient
from src.fl.baseserver import FedAvgAggregator

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

def make_dummy_loader(n=10):
    xs = torch.randn(n, 4)
    ys = torch.randint(0, 2, (n,))
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader(TensorDataset(xs, ys), batch_size=4)

def test_local_train_and_returned_structure():
    model = TinyModel()
    trainloader = make_dummy_loader(20)
    testloader = make_dummy_loader(10)
    client = BenignClient(id=0, trainloader=trainloader, testloader=testloader, model=model, lr=0.01, weight_decay=0.0, epochs=1)
    out = client.local_train(epochs=1, round_idx=0)
    assert 'weights' in out and 'num_samples' in out and 'client_id' in out and 'metrics' in out

def test_fedavg_aggregation_weighted_average():
    m1 = TinyModel()
    m2 = TinyModel()
    # set deterministic weights
    for p in m1.parameters():
        p.data.fill_(1.0)
    for p in m2.parameters():
        p.data.fill_(3.0)

    agg = FedAvgAggregator(model=TinyModel())
    # client 1 has 1 sample, client 2 has 3 samples -> averaged weight should be (1*1 + 3*3)/4 = 2.5
    agg.receive_update({k: v.cpu().clone() for k, v in m1.state_dict().items()}, length=1)
    agg.receive_update({k: v.cpu().clone() for k, v in m2.state_dict().items()}, length=3)
    averaged = agg.aggregate()
    # check first param tensor value
    first_key = list(averaged.keys())[0]
    val = averaged[first_key].flatten()[0].item()
    assert abs(val - 2.5) < 1e-6
