from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import copy as cp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseClient(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def num_samples(self) -> int:
        pass

    @abstractmethod
    def set_params(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def local_evaluate(self) -> Dict[str, Any]:
        pass


class BenignClient(BaseClient):
    def __init__(
        self,
        id: int,
        trainloader: Optional[DataLoader],
        testloader: Optional[DataLoader],
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int = 1,
        device: Optional[torch.device] = None,
    ):
        self._id = id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device if device is not None else torch.device("cpu")
        self.epochs_default = epochs
        self.lr = lr
        self.weight_decay = weight_decay

        # canonical attribute name
        self._model = model.to(self.device)
        self.dataset_len = len(trainloader.dataset) if trainloader is not None else 0

        # create optimizer bound to current model parameters
        self._create_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)
        self.loss_fn = nn.CrossEntropyLoss()

    # property accessor so code can use self.model
    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def _create_optimizer(self) -> None:
        # recreate optimizer (used at init and after loading new params)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)

    def get_id(self) -> int:
        return self._id

    def num_samples(self) -> int:
        return self.dataset_len

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        # load params and ensure model lives on device
        # accept cpu-state-dicts too
        self._model.load_state_dict({k: v.to(self.device) for k, v in params.items()})
        # re-init optimizer to avoid optimizer/param mismatch
        self._create_optimizer()

    def get_params(self) -> Dict[str, torch.Tensor]:
        # return CPU tensors for safe transport
        return {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        """Train for `epochs` local epochs and return a dict with 'weights','num_samples','client_id','metrics'."""
        self._model.train()
        epoch_count = epochs if epochs is not None else self.epochs_default
        for _ in range(epoch_count):
            if self.trainloader is None:
                break
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._model(inputs)
                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
        # step scheduler once per local call (you may prefer per-epoch)
        try:
            self.scheduler.step()
        except Exception:
            pass

        # collect metrics by running local evaluation
        metrics = self.local_evaluate()['metrics']
        result = {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }
        return result

    def local_evaluate(self) -> Dict[str, Any]:
        """Return dict: {'client_id', 'num_samples', 'metrics': {'loss':..., 'accuracy':...}}"""
        self._model.eval()
        loss_sum, correct, total, iters = 0.0, 0, 0, 0
        valloader = self.testloader or self.trainloader
        if valloader is None:
            return {'client_id': self.get_id(), 'num_samples': 0, 'metrics': {'loss': float('nan'), 'accuracy': float('nan')}}
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self._model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == targets).sum().item()
                loss_sum += self.loss_fn(outputs, targets).item()
                total += targets.size(0)
                iters += 1
        loss_avg = (loss_sum / iters) if iters > 0 else float('nan')
        acc = (correct / total) if total > 0 else float('nan')
        return {'client_id': self.get_id(), 'num_samples': total, 'metrics': {'loss': loss_avg, 'accuracy': acc}}
