import os
from functools import partial
from time import time
from typing import Callable, List

import torch
from poutyne.framework import Experiment, EarlyStopping, ReduceLROnPlateau
from torch.optim import Adam

from .callbacks import TransformerLrSchedule
from ..data import DataloaderFactory
from ..data import Dataset, DataSplitter
from ..metrics import loss, accuracy


def train_model(train_device: torch.device,
                data: List,
                model: torch.nn.Module,
                collate_fn: Callable,
                train_ratio: float,
                valid_ratio: float,
                batch_size: int,
                initial_learning_rate: float,
                num_epochs: int,
                lr_scheduler_patience: int,
                early_stopping_patience: int,
                lr_warmup: bool,
                log_dir: str,
                saving_dir: str,
                num_dataloader_workers: int,
                current_seed: int,
                additional_training_callbacks: List = None,
                additional_test_callbacks: List = None,
                train_subset_size: int = None,
                **kwargs):
    train_set, valid_set, test_set = DataSplitter.split_data(
        data, train_ratio, valid_ratio)

    if train_subset_size is not None:
        if train_subset_size > len(train_set):
            raise ValueError(
                f"There aren't enough training samples for a subset of {train_subset_size}"
            )

        train_set = train_set[:train_subset_size]

    dataloader_factory = DataloaderFactory(num_dataloader_workers, collate_fn)
    train_loader, valid_loader, test_loader = dataloader_factory.create(
        Dataset(train_set), Dataset(valid_set), Dataset(test_set), batch_size)

    optimizer = Adam(model.parameters(), lr=initial_learning_rate)

    loss_fn = partial(loss, device=train_device)
    accuracy_fn = partial(accuracy, device=train_device)

    exp = Experiment(os.path.join(os.getcwd(), log_dir, saving_dir,
                                  "_" + str(len(train_set)),
                                  "_" + str(current_seed)),
                     model,
                     logging=False,
                     device=train_device,
                     optimizer=optimizer,
                     loss_function=loss_fn,
                     batch_metrics=[accuracy_fn])

    callbacks = []
    if lr_warmup:
        lr_scheduler = TransformerLrSchedule(num_training_steps=num_epochs,
                                             num_warmup_steps=num_epochs * 0.1)
    else:
        lr_scheduler = ReduceLROnPlateau(patience=lr_scheduler_patience)

        early_stopping = EarlyStopping(patience=early_stopping_patience)
        callbacks.append(early_stopping)

    if additional_training_callbacks is not None:
        callbacks.extend(additional_training_callbacks)

    s = time()
    exp.train(train_loader,
              valid_generator=valid_loader,
              epochs=num_epochs,
              lr_schedulers=[lr_scheduler],
              callbacks=callbacks)

    print(f'Training time: {(time() - s) / 60}')

    exp.test(test_loader, callbacks=additional_test_callbacks)
