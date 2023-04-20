import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassStatScores
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .e2e import FeatureExtractor


class MyDR(pl.LightningModule):
    def __init__(self, config):
        super(MyDR, self).__init__()
        self.config = config

        self.fe = FeatureExtractor(config.model.e2e_fe)
        self.fe.load_state_dict(torch.load(config.model.e2e_fe.weight))
        for param in self.fe.parameters():
            param.requires_grad = False
        self.fe.eval()

        self.ln = nn.utils.parametrizations.orthogonal(nn.Linear(config.model.e2e_fe.channel, config.model.target_dim, bias=False))
    
    def compute_NPL(self, score, reduce='mean'):
        loss = torch.exp(score)
        loss = -torch.log(torch.div(loss[:, 0], torch.clamp(torch.sum(loss, dim=-1), min=0.00000001)))
        if reduce == 'mean':
            return torch.mean(loss)
        elif reduce == 'sum':
            return torch.sum(loss)

    def forward(self, x, s):
        with torch.no_grad():
            x, a = self.fe.select_feature(x)

            n, m, _, _, _ = s.shape
            s = torch.reshape(s, (-1, 3, self.config.dataset.input_size, self.config.dataset.input_size))
            s, sa = self.fe.select_feature(s)
            s = torch.reshape(s, (n, m, self.config.model.e2e_fe.channel, self.config.model.e2e_fe.size))
            sa = torch.reshape(sa, (n, m, self.config.model.e2e_fe.size, 1))

            a = torch.reshape(a, (n, 1, 1, self.config.model.e2e_fe.size))
            att_scores = torch.matmul(sa, a)            # N, M, S, S
            
        x = self.ln(torch.transpose(x, -2, -1))         # N, S, d
        s = self.ln(torch.transpose(s, -2, -1))         # N, M, S, d

        # normalized for cosine similarity
        x = torch.div(x, torch.clamp(torch.norm(x, dim=2, keepdim=True), min=1e-10))
        s = torch.div(s, torch.clamp(torch.norm(s, dim=3, keepdim=True), min=1e-10))

        cos_scores = torch.matmul(s, torch.unsqueeze(torch.transpose(x, -2, -1), dim=1))        # N, M, S, S
        cos_scores = torch.where(att_scores > 0, cos_scores, -2.0)
        cos_scores, _ = torch.max(cos_scores, dim=2)            # N, M, S
        pos_scores, neg_scores = torch.split(cos_scores, [1, self.config.model.n_pair], dim=1)
        pos_scores = torch.where(pos_scores > -2.0, pos_scores, 2.0)
        pos_scores, _ = torch.min(pos_scores, dim=-1)
        neg_scores, _ = torch.max(neg_scores, dim=-1)
        cos_scores = torch.cat((pos_scores, neg_scores), dim=-1)

        return cos_scores

    def training_step(self, batch, batch_idx):
        x, s = batch
        out = self.forward(x, s)
        loss = self.compute_NPL(out)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, s = batch
        out = self.forward(x, s)
        loss = self.compute_NPL(out, reduce='sum')
        return loss
    
    def validation_epoch_end(self, validation_step_outputs):
        loss = 0
        count = 0
        for out in validation_step_outputs:
            loss += out
            count += 1
        loss /= (self.config.dataset.batch_size*count)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
    
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

