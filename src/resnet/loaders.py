from torch.utils import data


class MutableDataLoader(data.DataLoader):
    def __init__(self, *args, train_transform=None, eval_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def train(self):
        self.dataset.transform = self.train_transform

    def eval(self):
        self.dataset.transform = self.eval_transform
