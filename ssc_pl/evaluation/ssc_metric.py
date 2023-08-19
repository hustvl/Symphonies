import torch
from torchmetrics import Metric


class SSCMetrics(Metric):

    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        for metric in ('tp_sc', 'fp_sc', 'fn_sc'):
            self.add_state(metric, torch.tensor(0), dist_reduce_fx='sum')
        for metric in ('tps_ssc', 'fps_ssc', 'fns_ssc'):
            self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds = torch.argmax(preds['ssc_logits'], dim=1)
        target = target['target']
        mask = target != self.ignore_index

        tp, fp, fn = self._calculate_sc_scores(preds, target, mask)
        self.tp_sc += tp
        self.fp_sc += fp
        self.fn_sc += fn

        tp, fp, fn = self._calculate_ssc_scores(preds, target, mask)
        self.tps_ssc += tp
        self.fps_ssc += fp
        self.fns_ssc += fn

    def compute(self):
        if self.tp_sc != 0:
            precision = self.tp_sc / (self.tp_sc + self.fp_sc)
            recall = self.tp_sc / (self.tp_sc + self.fn_sc)
            iou = self.tp_sc / (self.tp_sc + self.fp_sc + self.fn_sc)
        else:
            precision, recall, iou = 0, 0, 0
        ious = self.tps_ssc / (self.tps_ssc + self.fps_ssc + self.fns_ssc + 1e-6)
        return {
            'Precision': precision,
            'Recall': recall,
            'IoU': iou,
            'iou_per_class': ious,
            'mIoU': ious[1:].mean()
        }

    def _calculate_sc_scores(self, preds, targets, nonempty=None):
        preds = preds.clone()
        targets = targets.clone()
        bs = preds.shape[0]

        mask = targets == self.ignore_index
        preds[mask] = 0
        targets[mask] = 0

        preds = preds.flatten(1)
        targets = targets.flatten(1)
        preds = torch.where(preds > 0, 1, 0)
        targets = torch.where(targets > 0, 1, 0)

        tp, fp, fn = 0, 0, 0
        for i in range(bs):
            pred = preds[i]
            target = targets[i]
            if nonempty is not None:
                nonempty_ = nonempty[i].flatten()
                pred = pred[nonempty_]
                target = target[nonempty_]
            pred = pred.bool()
            target = target.bool()

            tp += torch.logical_and(pred, target).sum()
            fp += torch.logical_and(pred, ~target).sum()
            fn += torch.logical_and(~pred, target).sum()
        return tp, fp, fn

    def _calculate_ssc_scores(self, preds, targets, nonempty=None):
        preds = preds.clone()
        targets = targets.clone()
        bs = preds.shape[0]
        C = self.num_classes

        mask = targets == self.ignore_index
        preds[mask] = 0
        targets[mask] = 0

        preds = preds.flatten(1)
        targets = targets.flatten(1)

        tp = torch.zeros(C, dtype=torch.int).to(preds.device)
        fp = torch.zeros(C, dtype=torch.int).to(preds.device)
        fn = torch.zeros(C, dtype=torch.int).to(preds.device)
        for i in range(bs):
            pred = preds[i]
            target = targets[i]
            if nonempty is not None:
                mask = nonempty[i].flatten() & (target != self.ignore_index)
                pred = pred[mask]
                target = target[mask]
            for c in range(C):
                tp[c] += torch.logical_and(pred == c, target == c).sum()
                fp[c] += torch.logical_and(pred == c, ~(target == c)).sum()
                fn[c] += torch.logical_and(~(pred == c), target == c).sum()
        return tp, fp, fn
