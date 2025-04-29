import os
import time
import shutil

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter

from . import sod_metric


class Averager:

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer:

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip("/"))
    if os.path.exists(path):
        if remove and (
            basename.startswith("_")
            or input("{} exists, remove? (y/[n]): ".format(path)) == "y"
        ):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, "tensorboard"))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return "{:.1f}M".format(tot / 1e6)
        else:
            return "{:.1f}K".format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {"sgd": SGD, "adam": Adam, "adamw": AdamW}[optimizer_spec["name"]]
    optimizer = Optimizer(param_list, **optimizer_spec["args"])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec["sd"])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    # if flatten:
    #     ret = ret.view(-1, ret.shape[-1])

    return ret


def calc_cod(y_pred, y_true):
    batchsize = y_true.shape[0]

    metric_FM = sod_metric.Fmeasure()
    metric_WFM = sod_metric.WeightedFmeasure()
    metric_SM = sod_metric.Smeasure()
    metric_EM = sod_metric.Emeasure()
    metric_MAE = sod_metric.MAE()
    with torch.no_grad():
        assert y_pred.shape == y_true.shape

        for i in range(batchsize):
            true, pred = (
                y_true[i, 0].cpu().data.numpy() * 255,
                y_pred[i, 0].cpu().data.numpy() * 255,
            )

            metric_FM.step(pred=pred, gt=true)
            metric_WFM.step(pred=pred, gt=true)
            metric_SM.step(pred=pred, gt=true)
            metric_EM.step(pred=pred, gt=true)
            metric_MAE.step(pred=pred, gt=true)

        fm = metric_FM.get_results()["fm"]
        wfm = metric_WFM.get_results()["wfm"]
        sm = metric_SM.get_results()["sm"]
        em = metric_EM.get_results()["em"]["curve"].mean()
        mae = metric_MAE.get_results()["mae"]

    return sm, em, wfm, mae


from sklearn.metrics import precision_recall_curve


def calc_f1(y_pred, y_true):
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1, auc = 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            true = true.astype(int)
            pred = y_pred[i].flatten()

            precision, recall, thresholds = precision_recall_curve(true, pred)

            # auc
            auc += roc_auc_score(true, pred)
            # auc += roc_auc_score(np.array(true>0).astype(np.int), pred)
            f1 += max(
                [(2 * p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)]
            )

    return f1 / batchsize, auc / batchsize, np.array(0), np.array(0)


def calc_fmeasure(y_pred, y_true):
    # here i want to favor recall over precision, since i want to minimise false negatives, the test ground truth could have missed some small shacks
    batchsize = y_true.shape[0]

    mae, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(batchsize):
            gt_float, pred_float = (
                y_true[i, 0].cpu().data.numpy(),
                y_pred[i, 0].cpu().data.numpy(),
            )

            # # MAE
            mae.append(
                np.sum(cv2.absdiff(gt_float.astype(float), pred_float.astype(float)))
                / (pred_float.shape[1] * pred_float.shape[0])
            )
            # mae.append(np.mean(np.abs(pred_float - gt_float)))
            #
            pred = np.uint8(pred_float * 255)
            gt = np.uint8(gt_float * 255)

            pred_float_ = np.where(
                pred > min(1.5 * np.mean(pred), 255),
                np.ones_like(pred_float),
                np.zeros_like(pred_float),
            )
            gt_float_ = np.where(
                gt > min(1.5 * np.mean(gt), 255),
                np.ones_like(pred_float),
                np.zeros_like(pred_float),
            )

            preds.extend(pred_float_.ravel())
            gts.extend(gt_float_.ravel())

        RECALL = recall_score(gts, preds)
        PERC = precision_score(gts, preds)

        beta_sq = 4

        fmeasure = (1 + beta_sq) * PERC * RECALL / (beta_sq * PERC + RECALL)
        MAE = np.mean(mae)

    return fmeasure, MAE, np.array(0), np.array(0)


from sklearn.metrics import roc_auc_score, recall_score, precision_score
import cv2


def calc_ber(y_pred, y_true):
    batchsize = y_true.shape[0]
    y_pred, y_true = y_pred.permute(0, 2, 3, 1).squeeze(-1), y_true.permute(
        0, 2, 3, 1
    ).squeeze(-1)
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        pos_err, neg_err, ber = 0, 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            pred = y_pred[i].flatten()

            TP, TN, FP, FN, BER, ACC = get_binary_classification_metrics(
                pred * 255, true * 255, 125
            )
            pos_err += (1 - TP / (TP + FN)) * 100
            neg_err += (1 - TN / (TN + FP)) * 100

    return (
        pos_err / batchsize,
        neg_err / batchsize,
        (pos_err + neg_err) / 2 / batchsize,
        np.array(0),
    )


def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = gt > threshold
        pred = pred > threshold
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return TP, TN, FP, FN, BER, ACC


def cal_ber(tn, tp, fn, fp):
    return 0.5 * (fp / (tn + fp) + fn / (fn + tp))


def cal_acc(tn, tp, fn, fp):
    return (tp + tn) / (tp + tn + fp + fn)
