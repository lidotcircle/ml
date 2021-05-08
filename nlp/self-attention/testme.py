import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math


class Toy(nn.Module):
    def __init__(self, input: int, output: int, device: str = "cpu", expansion: int = 2):
        super(Toy, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input, input * expansion).to(device),
            nn.ReLU().to(device),
            nn.Linear(input* expansion, output).to(device)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def functionToLearn(input: torch.Tensor) -> torch.Tensor:
    sum = torch.einsum("i->", [input])
    return torch.tensor([sum])
    # return torch.tensor([math.sin(sum)])

TEST_INPUT_SIZE = 1024
TEST_DATA_NO = 10000
TEST_BATCH_SIZE = 4096
TEST_EPCHO = 1000
LEARNING_RATE = 0.003

class ToyDataset(Dataset):
    def __init__(self, input_size: int, dataset_size: int, device: str):
        super(ToyDataset).__init__()
        self.device = device
        self.dataset = list(
            map(
                lambda x: (x, functionToLearn(x).to(device)), 
                [ torch.randn(input_size).to(device) for i in range(dataset_size) ]
            )
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    model = Toy(TEST_INPUT_SIZE, 1, device)
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.L1Loss()

    i = 0
    dataset = DataLoader(dataset = ToyDataset(TEST_INPUT_SIZE, TEST_DATA_NO, device), shuffle = True, batch_size = TEST_BATCH_SIZE)
    for epcho in range(TEST_EPCHO):
        for x, y in dataset:
            i = i + 1
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"device: {device}, epcho: {epcho}, batch: {i}, loss: {loss:>7f}")
    
    for test in range(10):
        x = torch.randn(TEST_INPUT_SIZE).to(device)
        y = functionToLearn(x).to(device)
        pred = model(x)
        print(f"y: {y}, pred: {pred}")
