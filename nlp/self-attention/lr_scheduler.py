# learning rate schedulers
import math


class DampeningCos():
    def __init__(self, min_lr: int, max_lr: int, dampen: float, cycle: int = 1):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.dampen = dampen
        self.cycle  = cycle
        self.gone = 0

    def __call__(self, epoch: int = None):
        if epoch is None:
            self.gone = self.goen + 1
        else:
            self.gone = epoch
        ratio = self.gone / self.cycle
        gone_cycle = math.floor(ratio)
        phase = ratio - gone_cycle
        max_lr = self.max_lr * (self.dampen ** gone_cycle)
        min_lr = self.min_lr
        if max_lr <= min_lr:
            return min_lr
        amplitude = max_lr - min_lr
        res = (amplitude / 2) * math.cos(phase * 2 * math.pi) + amplitude / 2 + min_lr
        return res
