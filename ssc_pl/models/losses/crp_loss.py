import torch.nn.functional as F


def context_relation_loss(pred, target):
    pred_logits = pred['P_logits'].float()
    CP_mega_matrices = target['CP_mega_matrix']
    logits, labels = map(
        lambda x: x.flatten(2).transpose(1, 2).flatten(end_dim=1),
        (pred_logits.transpose(-1, -2), CP_mega_matrices))  # bs * N * n_mega_vox, n_rel

    cnt_neg = (labels == 0).sum(0)
    cnt_pos = labels.sum(0)
    pos_weight = cnt_neg / cnt_pos
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
