from pathlib import Path
import numpy
import math
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import TextClassificationDataset
from model import TextClassificationModel
from torch import nn
from torch.utils.data import DataLoader


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def load_model(model: nn.Module):
    file = Path("model.pth")
    if file.is_file():
        print("load model")
        model.load_state_dict(torch.load(file))

def save_model(model: nn.Module, postfix: str = ''):
    print("save model")
    torch.save(model.state_dict(), f"model{postfix}.pth")


def train(model: nn.Module, dataset: TextClassificationDataset, optimizer: torch.optim.Optimizer,
          epoch: int = 1000, batch_size: int = 300, shuffle: bool = True, device: str = "cpu"):
    dataloader = DataLoader(dataset, dataset.sampler(batch_size, epoch, shuffle), num_workers=2)
    loss_fn = nn.CrossEntropyLoss()
    sample_count = 0
    batchs_count = 0
    loss_list = [ ]
    loss_weight = []
    unsaved_count = 0
    for sentence, label in dataloader:
        batchs_count = batchs_count + 1
        sentence = sentence.to(device)
        label = label.to(device)
        bs = sentence.shape[0]
        sample_count = sample_count + bs
        unsaved_count = unsaved_count + bs
        ep = (sample_count - 1) // len(dataset)

        loss = loss_fn(pred, label)
        pred = model(sentence)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(float(loss))
        loss_weight.append(bs)
        if unsaved_count > len(dataset) * 0.25:
            loss_mean, loss_std = weighted_avg_and_std(loss_list, loss_weight)
            loss_list = []
            loss_weight = []
            unsaved_count = 0
            save_model(model)
            print(f"device: {device}, epoch: {ep}, batch size: {batch_size}, batch: {batchs_count}, total sample: {sample_count}",
                  f"loss mean: {loss_mean:>7f}, loss_std: {loss_std:>7f}")

        if sample_count % (5 * len(dataset)) == 0:
            eval(model, dataset, batch_size)


def eval(model: nn.Module, dataset: TextClassificationDataset, batch_size: int, device: str):
    loss_fn = nn.CrossEntropyLoss()
    dataset.toggleEvalMode()
    dataloader = DataLoader(dataset, dataset.sampler(batch_size, 1, False))

    loss_list = [ ]
    loss_weight = [ ]
    for sentence, label in dataloader:
        size = label.shape[0]
        sentence = sentence.to(device)
        label = label.to(device)
        pred = model(sentence)
        loss = loss_fn(pred, label)
        loss_list.append(float(loss))
        loss_weight.append(size)

    loss_mean, loss_std = weighted_avg_and_std(loss_list, loss_weight)
    print(f"eval in test dataset, loss_mean: {loss_mean}, loss_std: {loss_std}")

    dataset.toggleEvalMode()


device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    model = TextClassificationModel()
    dataset = TextClassificationDataset()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07, momentum=0.95)

    train(model, dataset, optimizer)
