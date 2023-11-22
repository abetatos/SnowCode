import torch

def MSE(y_true, y_pred, mask):
    delta = torch.pow(torch.pow(y_true-y_pred, 2), 2)
    masked_delta = torch.multiply(delta, mask)
    return torch.sqrt(masked_delta.nanmean())

def MAE(y_true, y_pred, mask):
    delta = torch.abs(y_true-y_pred)
    masked_delta = torch.multiply(delta, mask)
    return masked_delta.nanmean()
