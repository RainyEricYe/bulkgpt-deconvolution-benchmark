"""
Sweetwater model: interpretable autoencoder for tissue deconvolution.
Adapted from https://github.com/ML4BM-Lab/Sweetwater
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import models_utils


class SweetWater:
    def __init__(self, data, bulkrna, name, batch_size=128, epochs=1000,
                 lr=0.01, verbose=False, earlystopping=True):
        self.verbose = verbose
        self.xtrain, self.ytrain, self.xtest, self.ytest = data
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Split bulk for early stopping during phase 2
        bulk_np = bulkrna if isinstance(bulkrna, np.ndarray) else bulkrna
        print(f"    SweetWater: bulk_np type={type(bulk_np).__name__}, shape={bulk_np.shape if hasattr(bulk_np, 'shape') else '?'}")
        self.bulk_train, self.bulk_test = train_test_split(
            bulk_np, test_size=0.2, random_state=13
        )
        print(f"    SweetWater: bulk_train shape={self.bulk_train.shape if hasattr(self.bulk_train, 'shape') else '?'}")
        self.bulk_train = torch.tensor(self.bulk_train).float()
        self.bulk_test = torch.tensor(self.bulk_test).float().to(self.device)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.earlystopping = earlystopping
        if self.earlystopping:
            self.p1es = models_utils.EarlyStopper(patience=10)
            self.p2es = models_utils.EarlyStopper(patience=10)
            self.p3es = models_utils.EarlyStopper(patience=50)
        self.name = name
        self.setup()

    def setup(self):
        self.aemodel = models_utils.SweetWaterAutoEncoder(
            num_features=self.xtrain.shape[1],
            num_classes=self.ytrain.shape[1]
        ).to(self.device)
        self.mseloss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.aemodel.parameters(), lr=self.lr)

        # Phase 1: Pseudo-bulk alignment (autoencoder)
        self.phase1_ds = models_utils.SingleDataset(self.xtrain)
        self.phase1_dl = models_utils.DataLoader(
            self.phase1_ds, batch_size=self.batch_size, shuffle=True
        )
        self.phase1_ds_test = models_utils.SingleDataset(self.xtest)
        self.phase1_dl_test = models_utils.DataLoader(
            self.phase1_ds_test, batch_size=self.batch_size, shuffle=True
        )

        # Phase 2: Bulk alignment (autoencoder)
        self.phase2_ds = models_utils.SingleDataset(self.bulk_train)
        self.phase2_dl = models_utils.DataLoader(
            self.phase2_ds, batch_size=1, shuffle=True
        )

        # Phase 3: Proportion deconvolution
        self.phase3_ds = models_utils.PairedDataset(self.xtrain, self.ytrain)
        self.phase3_dl = models_utils.DataLoader(
            self.phase3_ds, batch_size=self.batch_size, shuffle=True
        )
        self.phase3_ds_test = models_utils.PairedDataset(self.xtest, self.ytest)
        self.phase3_dl_test = models_utils.DataLoader(
            self.phase3_ds_test, batch_size=self.batch_size, shuffle=True
        )

        self.r2 = lambda true, pred: 1 - (
            (np.square((true - pred)).mean()) /
            (np.square(true - true.mean(axis=0))).mean()
        )

    def train(self, x, ytrue=None, mode='phase1'):
        self.optimizer.zero_grad()
        if mode != 'phase3':
            xhat = self.aemodel(x, mode)
            loss = self.mseloss(x, xhat)
        else:
            yhat = self.aemodel(x, mode)
            loss = self.mseloss(ytrue, yhat)
        loss.backward()
        self.optimizer.step()
        if mode != 'phase3':
            return loss.item()
        else:
            return loss.item(), yhat

    @torch.no_grad()
    def test(self, x, ytrue=None, mode='phase1'):
        if mode != 'phase3':
            xhat = self.aemodel(x, mode)
            loss = self.mseloss(x, xhat)
        else:
            yhat = self.aemodel(x, mode)
            loss = self.mseloss(ytrue, yhat)
        if mode != 'phase3':
            return loss.item()
        else:
            return loss.item(), yhat

    def run(self):
        def run_phase1():
            aepseudotrloss, aepseudoteloss = [], []
            pbar = tqdm(total=int(self.epochs), mininterval=10, disable=not self.verbose)
            for _ in range(int(self.epochs)):
                btrloss, bteloss = [], []
                for xpseudo in self.phase1_dl:
                    loss_val = self.train(x=xpseudo.to(self.device), mode='phase1')
                    btrloss.append(loss_val)
                for xpseudo_test in self.phase1_dl_test:
                    bteloss.append(self.test(x=xpseudo_test.to(self.device), mode='phase1'))
                test_loss = np.mean(bteloss) if bteloss else 0
                aepseudotrloss.append(np.mean(btrloss) if btrloss else 0)
                aepseudoteloss.append(test_loss)
                if self.earlystopping and self.p1es.early_stop(test_loss):
                    if self.verbose:
                        print("Phase 1 early stopping")
                    break
                if self.verbose:
                    pbar.set_description(
                        f'P1: Train MSE {round(aepseudotrloss[-1], 6)}, '
                        f'Test MSE {round(aepseudoteloss[-1], 6)}'
                    )
                    pbar.update(1)

        def run_phase2():
            aebulktrloss, aebulkteloss = [], []
            pbar = tqdm(total=self.epochs, mininterval=10, disable=not self.verbose)
            for _ in range(self.epochs):
                btrloss = []
                for xbulk in self.phase2_dl:
                    loss_val = self.train(x=xbulk.to(self.device), mode='phase2')
                    btrloss.append(loss_val)
                aebulktrloss.append(np.mean(btrloss) if btrloss else 0)
                teloss = self.test(self.bulk_test, mode='phase2')
                aebulkteloss.append(teloss)
                if self.earlystopping and self.p2es.early_stop(teloss):
                    if self.verbose:
                        print("Phase 2 early stopping")
                    break
                if self.verbose:
                    pbar.set_description(
                        f'P2: Train MSE {round(aebulktrloss[-1], 6)}, '
                        f'Test MSE {round(aebulkteloss[-1], 6)}'
                    )
                    pbar.update(1)

        def run_phase3():
            pbar = tqdm(total=self.epochs, mininterval=10, disable=not self.verbose)
            for _ in range(self.epochs):
                batched_trproploss = []
                for xpseudo, bypseudo in self.phase3_dl:
                    bproploss, _ = self.train(
                        x=xpseudo.to(self.device),
                        ytrue=bypseudo.to(self.device),
                        mode='phase3'
                    )
                    batched_trproploss.append(bproploss)

                btestproploss = []
                for xpseudo_test, bypseudo_test in self.phase3_dl_test:
                    testproploss_, _ = self.test(
                        xpseudo_test.to(self.device),
                        bypseudo_test.to(self.device),
                        mode='phase3'
                    )
                    btestproploss.append(testproploss_)

                test_loss = np.mean(btestproploss) if btestproploss else 0
                if self.earlystopping and self.p3es.early_stop(test_loss):
                    if self.verbose:
                        print("Phase 3 early stopping")
                    break
                if self.verbose:
                    pbar.set_description(
                        f'P3: Train MSE {round(np.mean(batched_trproploss), 6)}, '
                        f'Test MSE {round(test_loss, 6)}'
                    )
                    pbar.update(1)

        # Freeze proportion layers, run phase 1 (pseudo-bulk autoencoder alignment)
        for name, param in self.aemodel.named_parameters():
            if 'prop' in name:
                param.requires_grad = False
        run_phase1()

        # Run phase 2 (bulk autoencoder alignment)
        run_phase2()

        # Freeze decoder, unfreeze proportion layers, run phase 3
        for name, param in self.aemodel.named_parameters():
            if 'decl' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        run_phase3()
