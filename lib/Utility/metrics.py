class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Evaluates a model's top k accuracy
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def caculate_metrics(tp, tn, fp, fn):
    """
    Evaluates the metrics
    """
    eps = 1e-8
    return {"total_accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "known_accuracy": tn / (tn + fp + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "f1": 2 * tp / (2 * tp + fp + fn + eps)}
