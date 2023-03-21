import torch
import torch.nn.functional as F


def ce_ssc_loss(pred, target):
    return F.cross_entropy(
        pred['ssc_logits'].float(),
        target['target'].long(),
        weight=target['class_weights'].float(),
        ignore_index=255,
        reduction='mean',
    )


def sem_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target['target']
    mask = target != 255
    target = target[mask]

    loss, cnt = 0, 0
    num_classes = pred.shape[1]
    for i in range(0, num_classes):
        p = pred[:, i]
        p = p[mask]
        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0

        if torch.sum(completion_target) > 0:
            cnt += 1.0
            nominator = (p * completion_target).sum()
            if p.sum() > 0:
                precision = nominator / p.sum()
                loss += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if completion_target.sum() > 0:
                recall = nominator / completion_target.sum()
                loss += F.binary_cross_entropy(recall, torch.ones_like(recall))
            if (1 - completion_target).sum() > 0:
                specificity = (((1 - p) * (1 - completion_target)).sum() /
                               (1 - completion_target).sum())
                loss += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
    return loss / cnt


def geo_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target['target']
    mask = target != 255

    empty_probs = pred[:, 0]
    nonempty_probs = 1 - empty_probs
    empty_probs = empty_probs[mask]
    nonempty_probs = nonempty_probs[mask]

    nonempty_target = target != 0
    nonempty_target = nonempty_target[mask].float()

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    specificity = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (F.binary_cross_entropy(precision, torch.ones_like(precision)) +
            F.binary_cross_entropy(recall, torch.ones_like(recall)) +
            F.binary_cross_entropy(specificity, torch.ones_like(specificity)))


def frustum_proportion_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)

    frustums_masks = target['frustums_masks']
    frustums_class_dists = target['frustums_class_dists']
    num_frustums = frustums_class_dists.shape[1]
    batch_cnt = frustums_class_dists.sum(0)  # n_fstm, n_cls

    frustum_loss = 0
    frustum_nonempty = 0
    for f in range(num_frustums):
        frustum_mask = frustums_masks[:, f].unsqueeze(1)
        prob = frustum_mask * pred  # bs, n_cls, H, W, D
        prob = prob.flatten(2).transpose(0, 1)
        prob = prob.flatten(1)  # n_cls, bs * H * W * D
        cum_prob = prob.sum(dim=1)  # n_cls

        total_cnt = batch_cnt[f].sum()
        total_prob = prob.sum()
        if total_prob > 0 and total_cnt > 0:
            fp_target = batch_cnt[f] / total_cnt
            cum_prob = cum_prob / total_prob

            nonzeros = fp_target != 0
            nonzero_p = cum_prob[nonzeros]
            frustum_loss += F.kl_div(torch.log(nonzero_p), fp_target[nonzeros], reduction='sum')
            frustum_nonempty += 1
    return frustum_loss / frustum_nonempty
