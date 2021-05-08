import torch
import torch.nn as nn
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


def sumSin(input: torch.Tensor) -> torch.Tensor:
    sum = torch.einsum("i->", [input])
    return torch.tensor([sum])
    # return torch.tensor([math.sin(sum)])

TEST_INPUT_SIZE = 1024
TEST_DATA_NO = 30000
TEST_BATCH_SIZE = 2048
TEST_EPCHO = 1000
LEARNING_RATE = 0.01
dataset = list(map(lambda x: (x, sumSin(x)), [ torch.randn(TEST_INPUT_SIZE) for i in range(TEST_DATA_NO) ]))

def makeBatch():
    start = 0
    end = len(dataset)
    while start < end:
        l = min(end - start, TEST_BATCH_SIZE)
        n = dataset[start:start + l]
        start = start + l
        ti = torch.stack(list(map(lambda xy: xy[0], n)), dim = 0)
        to = torch.stack(list(map(lambda xy: xy[1], n)), dim = 0)
        yield ti, to

device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == '__main__':
    model = Toy(TEST_INPUT_SIZE, 1, device)
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.L1Loss()

    i = 0
    for epcho in range(TEST_EPCHO):
        for x, y in makeBatch():
            i = i + 1
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"device: {device}, epcho: {epcho}, batch: {i}, loss: {loss:>7f}")
    
    for test in range(10):
        x = torch.randn(TEST_INPUT_SIZE).to(device)
        y = sumSin(x).to(device)
        pred = model(x)
        print(f"y: {y}, pred: {pred}")
