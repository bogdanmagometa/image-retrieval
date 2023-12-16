import torch
from torch.nn import functional as F


def calculate_metrics(logits, labels):
    probs = F.softmax(logits, dim=1)
    predicted = probs > 0.5

    acc = torch.mean(torch.sum(predicted * labels, dim=1) / torch.sum(predicted | labels, dim=1))
    precision = torch.mean(torch.sum(predicted * labels, dim=1) / torch.sum(labels, dim=1))
    recall = torch.mean(torch.sum(predicted * labels, dim=1) / torch.sum(predicted, dim=1))
    f1 = torch.mean(2 * torch.sum(predicted * labels, dim=1) / (torch.sum(predicted, dim=1) + torch.sum(labels, dim=1)))

    return acc, precision, recall, f1

if __name__ == "__main__":
    logits = torch.tensor([
        [10, 5],
        [5, 10]
        ], dtype=torch.float32)
    labels = torch.tensor([
        [0, 1],
        [1, 0]
    ])
    acc, precision, recall, f1 = calculate_metrics(logits, labels)
    print(f"acc = {acc}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1 = {f1}")
