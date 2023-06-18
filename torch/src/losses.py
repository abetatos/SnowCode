from torchmetrics import StructuralSimilarityIndexMeasure
import torch

def SSIM(y_true, y_pred, mask):
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(y_true, y_pred)

def MSE(y_true, y_pred, mask):
    delta = torch.pow(y_true-y_pred, 2)
    masked_delta = torch.multiply(delta, mask)
    return masked_delta.nanmean()

def MQE(y_true, y_pred, mask):
    delta = torch.pow(y_true-y_pred, 4)
    masked_delta = torch.multiply(delta, mask)
    return masked_delta.nanmean()

def MAE(y_true, y_pred, mask):
    delta = torch.abs(y_true-y_pred)
    masked_delta = torch.multiply(delta, mask)
    return masked_delta.nanmean()

def MDE(y_true, y_pred, mask):
    delta = torch.abs(y_pred/(y_true+ 1e-10) - 1)
    masked_delta = torch.multiply(delta, mask)
    return masked_delta.nanmean()

def MAEL(y_true, y_pred, mask):
    delta = torch.abs(y_true-y_pred)/(y_true + 1e-10)
    masked_delta = torch.multiply(delta, mask)
    return masked_delta.nanmean()

