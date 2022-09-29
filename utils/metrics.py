import torch





class BaseMetrics:
    def __init__(self, name):
        self.name = name
        self.training_loss = []

class Metrics:
    def __init__(self):
        pass

    def update(self):
        pass

class Recorder:
    def __init__(self):
        pass

    def get_fold_metrics(self, fold):
        pass


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res