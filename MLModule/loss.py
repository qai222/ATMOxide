import torch


class CELoss(torch.nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
        return -(y_true * torch.log(y_pred)).sum(dim=1).mean()
