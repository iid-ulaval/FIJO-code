import os
from random import shuffle

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf
from poutyne.utils import set_seeds
from torch import device, cuda

from .data.collator import EmbeddingCollator, TokenCollator
from .data.embedding import FasttextVectorizer, CamembertTokenizer
from .model import CamembertClassifier, LstmClassifier
from .training import train_model
from .training.callbacks import WandbLogger
from .utils import prepare_raw_data


@hydra.main(config_path="../conf", config_name="config")
def main(conf):
    for seed in conf["training"]["hyperparams"]["seeds"]:

        data, tags_to_idx = prepare_raw_data(
            os.path.join(get_original_cwd(), conf["data"]["path_to_data"]))

        if conf["training"]["hyperparams"]["shuffle_seed"] is not None:
            shuffle(data,
                    lambda: conf["training"]["hyperparams"]["shuffle_seed"])

        train_device = device(
            f'cuda:{conf["training"]["hyperparams"]["train_device_id"]}'
            if cuda.is_available() else 'cpu')

        subset_sizes = conf["training"]["hyperparams"][
            "train_data_subset_sizes"]

        for subset_size in ([i for i in subset_sizes]
        if len(subset_sizes) > 0 else [None]):
            set_seeds(seed)

            model = instantiate(conf["model"], output_size=len(tags_to_idx))

            collator = None
            if isinstance(model, LstmClassifier):
                embedding_model = FasttextVectorizer(
                    os.path.join(get_original_cwd(),
                                 conf["embedding"]["embeddings_path"]),
                    tags_to_idx)

                collator = EmbeddingCollator(embedding_model)
            elif isinstance(model, CamembertClassifier):
                tokenizer = CamembertTokenizer(tags_to_idx)

                collator = TokenCollator(tokenizer)
            else:
                raise NotImplementedError(
                    f"There's no model of type: {type(model)}")

            dict_conf = OmegaConf.to_container(conf)

            dict_conf["training"]["hyperparams"][
                "run_train_data_subset_size"] = subset_size

            dict_conf["training"]["hyperparams"]["seed"] = seed

            logger = WandbLogger(
                conf["training"]["logs"]["logger"]["projet_name"],
                conf["training"]["logs"]["logger"]["group_name"], dict_conf)

            train_model(train_device=train_device,
                        data=data,
                        model=model,
                        collate_fn=collator.collate_batch,
                        train_subset_size=subset_size,
                        additional_training_callbacks=[logger],
                        additional_test_callbacks=[logger],
                        **conf["training"]["hyperparams"],
                        **conf["training"]["logs"]["local"],
                        current_seed=seed,
                        tags_to_idx=tags_to_idx)


if __name__ == "__main__":
    main()
