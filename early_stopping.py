import numpy as np
import torch

class EarlyStopping:
    """Klasa za zaustavljanje treniranja ranije na osnovu performansi validacije."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): Koliko epoha treba čekati posle zadnjeg najboljeg unapređenja.
                            Default: 7
            verbose (bool): Ako je True, štampa poruke za svako unapređenje. Default: False
            delta (float): Minimalno unapređenje potrebno za smanjenje greške. Default: 0
            path (str): Putanja gde će model biti sačuvan. Default: 'checkpoint.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Sačuvaj model kada je došlo do unapređenja validacione greške.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
