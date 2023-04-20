import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from models.dr_train import MyDR
from utils import convert_dict_to_tuple
from data_utils.get_dataloader import get_dataloaders


if __name__ == '__main__':
    config = convert_dict_to_tuple(yaml.safe_load(open('MCS2023_baseline/config/exp9.yml', 'r')))

    model = MyDR(config)

    train_loader, valid_loader = get_dataloaders(config)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min")
    logger = TensorBoardLogger(save_dir=config.outdir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=config.outdir, filename='dr_exp9_resume_{epoch}_{val_loss:.4f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(accelerator=device, devices=1, logger=logger, callbacks=[early_stop_callback, checkpoint_callback, lr_monitor], max_epochs=200, inference_mode=False, log_every_n_steps=10, check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, valid_loader, ckpt_path='experiments/dr/dr_exp9_epoch=15_val_loss=1.2882.ckpt')
