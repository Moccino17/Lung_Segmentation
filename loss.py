import torch


class DiceLoss(torch.nn.Module):
    def init(self):
        super(DiceLoss, self).init()

    def forward(self, pred, target):
        intersection = pred * target
        tp = torch.sum(intersection)
        fp = torch.sum(pred - intersection)
        fn = torch.sum(target - intersection)
        return 1 - 2 * tp / (2 * tp + fn + fp)


def accuracy_score(pred, target):
    intersection = pred * target
    tp = torch.sum(intersection).data.item()
    fp = torch.sum(pred - intersection).data.item()
    fn = torch.sum(target - intersection).data.item()
    n = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3]
    tn = n - fp - fn - tp
    return (tp + tn) / (tp + tn + fp + fn) * 100


def dice_score(pred, target):
    intersection = pred * target
    tp = torch.sum(intersection).data.item()
    fp = torch.sum(pred - intersection).data.item()
    fn = torch.sum(target - intersection).data.item()
    return 100 * 2 * tp / (2 * tp + fp + fn)


def jaccard_index(pred, target):
    intersection = pred * target
    tp = torch.sum(intersection).data.item()
    fp = torch.sum(pred - intersection).data.item()
    fn = torch.sum(target - intersection).data.item()
    return tp / (tp + fp + fn) * 100


def sensitivity_score(pred, target):
    intersection = pred * target
    tp = torch.sum(intersection).data.item()
    fn = torch.sum(target - intersection).data.item()
    return tp / (tp + fn) * 100


def specificity_score(pred, target):
    intersection = pred * target
    tp = torch.sum(intersection).data.item()
    fn = torch.sum(target - intersection).data.item()
    fp = torch.sum(pred - intersection).data.item()
    n = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3]
    tn = n - fp - fn - tp
    return tn / (tn + fp) * 100
