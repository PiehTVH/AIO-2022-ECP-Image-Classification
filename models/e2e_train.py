import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassStatScores
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm


class E2EFeatureExtractor(pl.LightningModule):
    def __init__(self, config):
        super(E2EFeatureExtractor, self).__init__()
        self.config = config
        self.fe = timm.create_model(
                                    config.model.pt_weight,
                                    pretrained=True,
                                    num_classes=0,  # remove classifier nn.Linear
                                    )
        self.mlp = nn.Sequential(
            nn.Linear(config.model.channel, config.model.channel),
            nn.LeakyReLU(),
            nn.Linear(config.model.channel, 1),
            )
        self.cls = nn.Linear(config.model.channel, config.dataset.num_classes, bias=False)
        self.stats = MulticlassStatScores(config.dataset.num_classes, average=None)
    
    def compute_arcloss(self, pred, target, reduce='mean'):
        angle = torch.acos(torch.clamp(pred, min=-1.0, max=1.0))
        angle[torch.arange(pred.size(0)), target] += self.config.model.arcloss.m
        angle = torch.cos(angle)
        loss = -torch.log(torch.exp(self.config.model.arcloss.s*angle[torch.arange(pred.size(0)), target])/ torch.sum(torch.exp(self.config.model.arcloss.s*angle), dim=1))
        if reduce == 'mean':
            return torch.mean(loss)
        if reduce == 'sum':
            return torch.sum(loss)
    
    def compute_NPL(self, score):
        loss = torch.exp(score)
        loss = -torch.log(torch.div(loss[:, 0], torch.clamp(torch.sum(loss, dim=-1), min=0.00000001)))
        return torch.mean(loss)
    
    def forward(self, x, s=None):
        x = self.fe.forward_features(x)                 # N, C, H, W
        x = torch.flatten(x, 2)                         # N, C, S = H * W
        x = torch.div(x, torch.clamp(torch.norm(x, dim=1, keepdim=True), min=0.00000001))

        a = self.mlp(torch.transpose(x, 1, 2))           # N, S, 1
        a = F.softmax(a, dim=1)

        if s is not None:
            with torch.no_grad():
                n, m, _, _, _ = s.shape
                s = torch.reshape(s, (-1, 3, self.config.dataset.input_size, self.config.dataset.input_size))
                s = self.fe.forward_features(s)
                s = torch.reshape(s, (n, m, self.config.model.channel, self.config.model.size))
                s = torch.div(s, torch.clamp(torch.norm(s, dim=2, keepdim=True), min=0.00000001))
        
            score = torch.matmul(torch.transpose(s, 2, 3), torch.unsqueeze(x, dim=1))           # N, M, S, S
            score, _ = torch.max(score, dim=2)
            score = torch.matmul(score, a)
            score = torch.squeeze(score, dim=2)
        
        x = torch.matmul(x, a)
        x = torch.squeeze(x, dim=2)
        x = torch.div(x, torch.clamp(torch.norm(x, dim=1, keepdim=True), min=0.00000001))

        with torch.no_grad():
            self.cls.weight.div_(torch.clamp(torch.norm(self.cls.weight, dim=1, keepdim=True), min=0.00000001))
        
        x = self.cls(x)

        if s is None:
            return x
        else:
            return x, score
    
    def training_step(self, batch, batch_idx):
        x, y, s = batch
        out, score = self.forward(x, s)
        cls_loss = self.compute_arcloss(out, y)
        con_loss = self.compute_NPL(score)
        loss = self.config.model.w_cls * cls_loss + self.config.model.w_con * con_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('cls_loss', cls_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('con_loss', con_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.compute_arcloss(out, y, reduce='sum')
        stats = self.stats(F.softmax(out), y)
        return loss, stats
    
    def validation_epoch_end(self, validation_step_outputs):
        loss = 0
        count = 0
        stats = torch.zeros((self.config.dataset.num_classes, 5)).to('cuda')
        for out in validation_step_outputs:
            loss += out[0]
            stats += out[1]
            count += 1
        loss /= (self.config.dataset.batch_size*count)
        f1 = torch.nanmean(2*stats[:, 0]/(2*stats[:, 0]+stats[:,1]+stats[:,3]))
        precision = torch.nanmean(stats[:, 0]/(stats[:, 0]+stats[:, 1]))
        recall = torch.nanmean(stats[:, 0]/stats[:, 4])
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=False, logger=True)
    
    def configure_optimizers(self):
        if self.config.train.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.learning_rate)
        if self.config.train.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.train.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode=self.config.train.lr_schedule.mode, patience=self.config.train.lr_schedule.patience)
        return ({
            "optimizer": optimizer,
            "lt_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            }
        })
    
