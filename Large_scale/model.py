import torch
from tqdm import tqdm
from utils import apply
import torch.optim as optim

def default(*args, **kwargs):
    pass

class ModelWrapper:
    def __init__(self, name, **kwargs):
        # print('Model name is', name)
        if name in ['mraea', 'rrea']:
            from rrea import TFModelWrapper
            self.tf = True
            self.model = TFModelWrapper(name, **kwargs)
        elif name == 'gcn-align':
            from gcn_align import GCNAlignWrapper
            self.tf = True
            self.model = GCNAlignWrapper(**kwargs)

    def __getattr__(self, item):
        SHARED_METHODS = ['update_trainset',
                          'update_devset',
                          'train1step',
                          'test_train_pair_acc',
                          'get_curr_embeddings',
                          'mraea_iteration'
                          ]
        if item in SHARED_METHODS:
            if self.tf:
                if hasattr(self.model, item):
                    return object.__getattribute__(self.model, item)
                return default
            else:
                return object.__getattribute__(self, '_' + item)
        else:
            return self.__getattribute__(item)

    def _update_trainset(self, pairs, append=False):
        self.train_pair = torch.from_numpy(pairs).to(self.device)

    def _update_devset(self, pairs, append=False):
        self.dev_pair = pairs.to(self.device)

    def default_sgd(self):
        if not hasattr(self, '_default_sgd'):
            self._default_sgd = optim.RMSprop(self.model.parameters(), lr=self.model.lr)
        return self._default_sgd

    def _train1step(self, epoch=75, sgd=None):
        if sgd is None:
            sgd = self.default_sgd()
        self.model.refresh_cache()
        for it in tqdm(range(epoch)):
            self.model.train1step(it, self.train_pair, sgd)

        self.model.run_test(self.dev_pair)

    def _test_train_pair_acc(self):
        pass

    def _get_curr_embeddings(self, device=None):
        if device is None:
            device = self.device
        return apply(lambda x: x.detach().to(device), *self.model.get_embed())