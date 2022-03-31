from typing import Dict
from poutyne.framework.callbacks.lr_scheduler import _PyTorchLRSchedulerWrapper
from transformers import get_linear_schedule_with_warmup


class TransformerLrSchedule(_PyTorchLRSchedulerWrapper):

    def __init__(self, num_training_steps, num_warmup_steps):
        super().__init__(torch_lr_scheduler=None)
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

    def on_train_begin(self, logs: Dict):
        self.scheduler = get_linear_schedule_with_warmup(
            self.model.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps)
