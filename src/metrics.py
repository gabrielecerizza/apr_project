from operator import itemgetter

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(scores, labels):
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    assert np.isclose(eer1, eer2), (eer1, eer2)

    return eer1


def compute_error_rates(scores, labels):
    """Creates a list of false-negative rates, a list of 
    false-positive rates and a list of decision thresholds 
    that give those error-rates.

    Taken from the official VoxCeleb Speaker Recognition 
    Challenge 2021 implementation at:
        https://github.com/clovaai/voxceleb_trainer
    """

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


def compute_min_dcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1):
    """Computes the minimum of the detection cost function.  
    The comments refer to equations in Section 3 of the NIST 2016 
    Speaker Recognition Evaluation Plan.

    Taken from the official VoxCeleb Speaker Recognition 
    Challenge 2021 implementation at:
        https://github.com/clovaai/voxceleb_trainer

    The default values for the parameters are taken from the
    NIST 2018 Speaker Recognition Evaluation Plan, AfV source
    type.
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
