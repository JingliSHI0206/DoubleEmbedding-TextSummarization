import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, lr):
        self.last_score = None
        self.decay_times = 0
        self.max_decay_times = 5
        self.lr = lr
        self.max_grad_norm = 10
        self.method = 'adam'
        self.lr_decay = 0.5
        self.start_decay_at = 5
        self.start_decay = False

    def step(self):
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self, score, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_score = score
        self.optimizer.param_groups[0]['lr'] = self.lr
