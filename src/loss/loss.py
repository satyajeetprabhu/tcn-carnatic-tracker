import torch
import torch.nn.functional as F

def loss_bce(beats_pred, beats_gt, downbeats_pred, downbeats_gt):
    """
    Binary Cross Entropy loss for beats and downbeats.
    """
    loss_beat = F.binary_cross_entropy(beats_pred, beats_gt)
    loss_downbeat = F.binary_cross_entropy(downbeats_pred, downbeats_gt)
    loss_total = loss_beat + loss_downbeat
    return loss_total, loss_beat, loss_downbeat


# Can be used for meter based weighting. for example, for 4/4 meter, beat_weight=1.0 and downbeat_weight=4.0
def loss_weighted(beats_pred, beats_gt, downbeats_pred, downbeats_gt, beat_weight=1.0, downbeat_weight=1.0):
    loss_beat = F.binary_cross_entropy(beats_pred, beats_gt)
    loss_downbeat = F.binary_cross_entropy(downbeats_pred, downbeats_gt)
    loss_total = beat_weight * loss_beat + downbeat_weight * loss_downbeat
    return loss_total, loss_beat, loss_downbeat


# This function computes the loss for a multitask model with relative weights based on the number of beat and downbeat annotations.
def loss_relative(beats_pred, beats_gt, downbeats_pred, downbeats_gt):
    beat_count = torch.clamp(beats_gt.sum(), min=1.0)
    downbeat_count = torch.clamp(downbeats_gt.sum(), min=1.0)
    total_count = beat_count + downbeat_count

    # Inverse the weights to give more importance to the less frequent class
    beat_weight = downbeat_count / total_count
    downbeat_weight = beat_count / total_count

    loss_beat = F.binary_cross_entropy(beats_pred, beats_gt)
    loss_downbeat = F.binary_cross_entropy(downbeats_pred, downbeats_gt)
    loss_total = beat_weight * loss_beat + downbeat_weight * loss_downbeat

    return loss_total, loss_beat, loss_downbeat

# Add code for masked loss

