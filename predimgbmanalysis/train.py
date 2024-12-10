import os
from time import sleep
import numpy as np
import logging

import torch
from torch.autograd import grad, Function
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

try:
    # Try importing from the new location (pytorch_lightning.utilities.combined_loader)
    from pytorch_lightning.utilities.combined_loader import CombinedLoader
except ImportError:
    # If the new import fails, use the old location (pytorch_lightning.trainer.supporters)
    from pytorch_lightning.trainer.supporters import CombinedLoader

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import pandas as pd

from pytorch_lightning.callbacks import TQDMProgressBar

import predimgbmanalysis.utils as utils
from predimgbmanalysis.models import models_dict
from predimgbmanalysis.get_toydata import load_data_fn, fetch_dataloader, dataset_dict
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class AfterTrainCheckpoint(pl.Callback):
    """
    Callback for saving the checkpoint after training finishes
    ref: https://github.com/Lightning-AI/lightning/discussions/11779
    """

    def __init__(self, save_dir=None):
        super().__init__()
        self.save_dir = save_dir

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logger.info("Saving final checkpoint...")
        # As we advance one step at end of training, we use `global_step - 1`
        final_checkpoint_name = f"checkpoints/final_step_{trainer.global_step}_epoch_{trainer.current_epoch}.ckpt"
        if self.save_dir is not None:
            final_checkpoint_name = os.path.join(self.save_dir, final_checkpoint_name)
        else:
            final_checkpoint_name = os.path.join(trainer.log_dir, final_checkpoint_name)

        trainer.save_checkpoint(final_checkpoint_name)


class GetSaliencyCallback(pl.Callback):
    def __init__(self, log_saliency_every_n_batch=None):
        super().__init__()
        self.log_saliency_every_n_batch = log_saliency_every_n_batch

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        # either only get saliency only for first batch or for every log_saliency_every_n_batch batch
        if self.log_saliency_every_n_batch != -1:
            check_batch_idx = (
                batch_idx
                if self.log_saliency_every_n_batch is None
                else (batch_idx % self.log_saliency_every_n_batch)
            )
            if check_batch_idx == 0:
                torch.set_grad_enabled(True)
                for env_idx, env_key in enumerate(batch.keys()):
                    x_e, y_e = batch[env_key]
                    x_e.requires_grad_()
                    error_e = pl_module.step((x_e, y_e), env_idx)

                    error_e.mean().backward()
                    saliency_map_env = x_e.grad.data[:2].abs()
                    value_range = (
                        torch.min(saliency_map_env),
                        torch.max(saliency_map_env),
                    )
                    saliency_map_env /= max(value_range[1], 1e-12)
                    # if x_e is a 3d image then only take one slice of the image
                    if len(x_e.shape) == 5:
                        slice = x_e.shape[-1] // 2
                        saliency_map_env = saliency_map_env[:, :, :, :, slice]
                        input_img = batch[env_key][0][:2][:, :, :, :, slice]

                    pl_module.logger.experiment.add_image(
                        f"Saliency_map_env_{env_idx}",
                        saliency_map_env,
                        global_step=pl_module.global_step,
                        dataformats="NCHW",
                    )  # CHW
                    pl_module.logger.experiment.add_image(
                        f"Input_img_env_{env_idx}",
                        utils.normalize_image(input_img),
                        global_step=pl_module.global_step,
                        dataformats="NCHW",
                    )
                trainer.model.zero_grad()
                torch.set_grad_enabled(False)


class ToyModelImgModule(pl.LightningModule):
    def __init__(
        self,
        model_type,
        model_params,
        mode,
        loader_params,
        val_loader_params,
        data_type,
        data_params,
        optimizer,
        optimizer_params,
        use_cuda,
        criterion=None,
        treatidx_env_mapping={0: "CG", 1: "EG"},
        k_fold_splits=None,
        fold_idx=None,
        no_validation=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = models_dict.get(model_type, "conv")(**model_params)
        if criterion is None:
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            self.criterion = getattr(torch.nn, criterion)(reduction="none")

        self.mode = mode
        self.model_params = model_params

        self.loader_params = loader_params
        self.val_loader_params = val_loader_params

        self.data_type = data_type
        self.data_params = data_params
        self.val_data_params = data_params.copy()
        if "transform" in self.val_data_params:
            if self.val_data_params["transform"] == "resize_randomspatial_transform":
                self.val_data_params["transform"] = "tensor_spatial_transform"
            elif self.val_data_params["transform"] == "randomspatial_transform":
                self.val_data_params["transform"] = "spatial_transform"
            elif self.val_data_params["transform"].startswith(
                "randomspatialpad"
            ) and self.val_data_params["transform"].endswith("CT"):
                self.val_data_params["transform"] = "pad_transform_CT"
        self.optimizer = optimizer
        self.lrscheduler_params = {}
        if optimizer_params.get("lrscheduler_params") is not None:
            self.lrscheduler_params = optimizer_params.get("lrscheduler_params")
            optimizer_params = dict(optimizer_params.copy())
            optimizer_params.pop("lrscheduler_params")
        self.optimizer_params = optimizer_params
        self.use_cuda = use_cuda
        self.treatidx_env_mapping = treatidx_env_mapping
        self.k_fold_splits = k_fold_splits
        self.fold_idx = fold_idx
        self.no_validation = no_validation
        self.save_hyperparameters()

    def forward(self, x_e, treat=None, **kwargs):
        if treat is not None:
            return self.model(x_e, treat=treat)
        else:
            return self.model(x_e, **kwargs)

    def step(self, batch, env_idx):
        x_e, y_e = batch
        if self.mode in ["mtl_loss"]:
            output = self.forward(x_e)
            error_e = self.criterion(output[env_idx].squeeze(), y_e)
        elif self.mode in ["standard"]:
            output = self.forward(x_e)
            error_e = self.criterion(output.squeeze(), y_e)
        else:
            logger.warning(f"Unknown training mode {self.mode}.")

        return error_e

    def training_step(self, batch, batch_idx):
        error = 0
        for env_idx, env_key in enumerate(
            [self.treatidx_env_mapping[0], self.treatidx_env_mapping[1]]
        ):
            if self.k_fold_splits is not None:
                env_key = env_key + "traincv"

            error_e = self.step(batch[env_key], env_idx)
            error += error_e.mean()
            self.log(
                f"Error/Train_{env_idx}_{env_key}",
                error_e.mean(),
            )
        self.log(
            "Error/Train_total",
            error,
        )
        return error

    def validation_step(self, batch, batch_idx):
        if self.no_validation:
            self.log(
                "Error/Val_total",
                0,
                on_epoch=True,
                on_step=False,
            )
        else:
            error = 0

            for env_idx, env_key in enumerate(
                [self.treatidx_env_mapping[0], self.treatidx_env_mapping[1]]
            ):
                if self.k_fold_splits is not None:
                    env_key = env_key + "traincv"
                else:
                    env_key = env_key + "val"
                error_e = self.step(batch[env_key], env_idx)
                error += error_e.mean()
                self.log(
                    f"Error/Val_{env_idx}_{env_key}",
                    error_e.mean(),
                    on_epoch=True,
                    on_step=False,
                )
            self.log(
                "Error/Val_total",
                error,
                on_epoch=True,
                on_step=False,
            )

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_params)
        elif self.optimizer == "SGD_ReduceLROnPlateau":
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_params)
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **self.lrscheduler_params
                ),  # ,factor=0.1, patience=100, min_lr=1e-6),
                "monitor": "Error/Val_total",
                "interval": "epoch",
                "frequency": 1,
            }
            # "monitor": "Error/Train_total"}
            return [optimizer], [lr_scheduler]
        elif self.optimizer == "SGD_StepLR":
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_params)
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, **self.lrscheduler_params
                ),  # step_size=750, factor=0.1, patience=100, min_lr=1e-6),
                "monitor": "Error/Val_total",
                "interval": "epoch",
                "frequency": 1,
            }
            # "monitor": "Error/Train_total"}
        elif self.optimizer == "SGD_MultiStepLR":
            optimizer = torch.optim.SGD(self.parameters(), **self.optimizer_params)
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, **self.lrscheduler_params
                ),  # milestones=[1000,1500], factor=0.1, patience=100, min_lr=1e-6),
                "monitor": "Error/Val_total",
                "interval": "epoch",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler]
        else:
            logger.warning(
                f"No optimizer named {self.optimizer}. Using default optimizer: Adam"
            )
        return optimizer

    def train_dataloader(self):
        logger.info(f"Getting training dataloaders.")
        dataloaders = {}

        for env in [self.treatidx_env_mapping[0], self.treatidx_env_mapping[1]]:
            if self.k_fold_splits is not None:
                env = env + "traincv"
            dataset = dataset_dict[self.data_type](
                env=env,
                train=True,
                treatidx_env_mapping=self.treatidx_env_mapping,
                **self.data_params,
            )
            if self.k_fold_splits is not None:
                # get length of dataset
                dataset_len = len(dataset)
                # split dataset into kfoldcv folds
                kfold = KFold(
                    n_splits=self.k_fold_splits, shuffle=True, random_state=42
                )
                train_indices, _ = list(kfold.split(np.arange(dataset_len)))[
                    self.fold_idx
                ]
                # get subset of dataset
                dl = DataLoader(
                    dataset,
                    shuffle=False,  # set to False to use SubsetRandomSampler
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                    **self.loader_params,
                )

            else:
                dl = DataLoader(
                    dataset,
                    shuffle=True,
                    **self.loader_params,
                    # **kwargs,
                )
            dataloaders[env] = dl

        return dataloaders

    def val_dataloader(self):
        # cf. fetch_dataloader()
        if self.no_validation:
            logger.info(f"Not using validation dataloaders.")
            return None
        else:
            logger.info(f"Getting validation dataloaders.")
            dataloaders = {}

            for env in [self.treatidx_env_mapping[0], self.treatidx_env_mapping[1]]:
                if self.k_fold_splits is not None:
                    env = env + "traincv"
                else:
                    env = env + "val"
                dataset = dataset_dict[self.data_type](
                    env=env,
                    train=False,
                    treatidx_env_mapping=self.treatidx_env_mapping,
                    **self.val_data_params,
                )
                if self.k_fold_splits is not None:
                    # get length of dataset
                    dataset_len = len(dataset)
                    # split dataset into kfoldcv folds
                    kfold = KFold(
                        n_splits=self.k_fold_splits, shuffle=True, random_state=42
                    )
                    _, val_indices = list(kfold.split(np.arange(dataset_len)))[
                        self.fold_idx
                    ]
                    # get subset of dataset
                    dl = DataLoader(
                        dataset,
                        shuffle=False,  # set to False to use SubsetRandomSampler
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            val_indices
                        ),
                        **self.val_loader_params,
                    )
                else:
                    dl = DataLoader(
                        dataset,
                        shuffle=True,
                        **self.val_loader_params,
                        # **kwargs,
                    )
                dataloaders[env] = dl
            combined_loader = CombinedLoader(dataloaders, mode="max_size_cycle")
            return combined_loader
        # return dataloaders

    def test_dataloader(self):
        # cf. fetch_dataloader()
        dataloaders = {}
        kwargs = (
            {"num_workers": 1, "pin_memory": True}
            if self.use_cuda
            else {"num_workers": 0, "pin_memory": False}
        )

        dl = DataLoader(
            dataset_dict[self.data_type](env="test", train=False, **self.data_params),
            shuffle=False,
            **self.val_loader_params,
            **kwargs,
        )

        return dl


@hydra.main(
    version_base="1.2",
    config_path="/absolute/path/to/configs",
    config_name="config_cmnist",
)
def main(cfg: DictConfig):
    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        work_dir = HydraConfig.get().run.dir
    else:
        work_dir = os.path.join(
            HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir
        )

    cfg_pl = cfg.get("pl_params")
    if not torch.cuda.is_available():
        logging.warning(f"CUDA is not available, using cpu.")

    logging.debug(
        f"Training with following config:\n{OmegaConf.to_yaml(cfg, resolve=True)}"
    )
    assert cfg_pl.get("mode") in [
        "mtl_loss",
        "standard",
    ]

    if cfg_pl.data_params.get("b") is None:
        cfg_pl.data_params.b = [
            0.0,
            0.0,
            cfg_pl.get("data_params").get("b_prog"),
            cfg_pl.get("data_params").get("b_pred"),
        ]
        logging.info(
            f"Parameter b not specified. Using b={cfg_pl.data_params.get('b')}."
        )

    if (cfg_pl.get("data_params").get("save_num_data_dir") is None) and cfg.get(
        "save_num_data"
    ):
        cfg_pl.data_params.save_num_data_dir = work_dir

    callbacks = []

    tb_logger = TensorBoardLogger(save_dir=work_dir)
    checkpoint_callback = ModelCheckpoint(
        save_last=False,
        every_n_epochs=cfg.get(
            "log_checkpoint_every_n_epochs",
            cfg.get("trainer_params").get("max_epochs", 100),
        ),
        save_on_train_epoch_end=True,
        filename="max_epochs-{epoch}-{step}",
        save_top_k=cfg.get("save_top_k", -1),
    )
    # get TQDM progress bar with lower refresh rate
    if cfg.get("progress_bar_refresh_rate", None) is not None:
        callbacks.append(
            TQDMProgressBar(refresh_rate=cfg.get("progress_bar_refresh_rate", 1))
        )
    after_train_checkpoint_callback = AfterTrainCheckpoint()
    get_saliency_callback = GetSaliencyCallback(
        log_saliency_every_n_batch=cfg.get("log_saliency_every_n_batch", None)
    )

    model = ToyModelImgModule(**cfg_pl)
    callbacks.append(checkpoint_callback)
    callbacks.append(get_saliency_callback)

    trainer = pl.Trainer(
        **cfg.get("trainer_params"),
        logger=tb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model)
    trainer.reset_train_dataloader()
    trainer.reset_val_dataloader()


# %% Main
if __name__ == "__main__":
    main()
