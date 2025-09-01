from src.fl.baseclient import BenignClient
from typing import Any, Dict

class MaliciousClient(BenignClient):
    def __init__(self, *args, poisoner=None, selector=None, trigger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.poisoner = poisoner
        self.selector = selector
        self.trigger = trigger

    def local_train(self, epochs: int, round_idx: int) -> Dict[str, Any]:
        # if poisoner present, delegate to poisoner
        if self.poisoner is not None:
            # pass global params if needed
            # assumes caller (server/main) calls client.set_params(global) before local_train
            return self.poisoner.poison_and_train(self, self.selector, self.trigger, epochs=epochs, round_idx=round_idx, global_params=self.get_params())
        # otherwise behave as benign client
        return super().local_train(epochs, round_idx)
