import os

import torch
from skorch import callbacks
from skorch.dataset import CVSplit
from skorch.net import NeuralNet
from torch import nn

from MLModule.loss import CELoss
from MLModule.saoto_callbacks import get_callbacks

this_dir = os.path.dirname(os.path.abspath(__file__))


class FnnModule(nn.Module):
    def __init__(
            self,
            d_in,
            d_out=4,
            h_size=10,
            h_layers=2,
            dropout=0.2,
            nonlin=nn.ReLU()
    ):
        super(FnnModule, self).__init__()
        self.h_layers = h_layers
        self.input_layer = nn.Linear(d_in, h_size)
        self.middle_layer = nn.Linear(h_size, h_size)
        self.output_layer = nn.Linear(h_size, d_out)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(d_out, elementwise_affine=False)

    def forward(self, X):
        X = X.float()
        X = self.input_layer(X)
        X = self.nonlin(X)
        for _ in range(self.h_layers):
            X = self.middle_layer(X)
            X = self.nonlin(X)
            X = self.dropout(X)
        X = self.output_layer(X)

        X = self.softmax(X)
        # X = self.layer_norm(X)

        return X


class AutoEncoder(nn.Module):
    def __init__(
            self,
            d_in,
            hlayer_sizes=(128, 64, 32, 16),
    ):
        super(AutoEncoder, self).__init__()

        encoder_layers = []
        encoder_layers.append(nn.Linear(d_in, hlayer_sizes[0]))
        for i in range(len(hlayer_sizes) - 1):
            encoder_layers.append(nn.ReLU(True))
            encoder_layers.append(nn.Linear(hlayer_sizes[i], hlayer_sizes[i + 1]))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in reversed(range(len(hlayer_sizes))):
            if i == 0:
                decoder_layers.append(nn.Linear(hlayer_sizes[i], d_in))
                decoder_layers.append(nn.ReLU(True))
            else:
                decoder_layers.append(nn.Linear(hlayer_sizes[i], hlayer_sizes[i - 1]))
                decoder_layers.append(nn.ReLU(True))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, X):
        """
        https://github.com/skorch-dev/skorch/issues/451
        """
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded


Fnn_kargs = dict(
    # hyperparams that should be tuned
    lr=5e-5,
    batch_size=128,
    module__h_size=10000,
    module__h_layers=1,
    module__dropout=0.3,

    # hyperparams that would not be tuned
    optimizer=torch.optim.Adam,
    predict_nonlinearity=None,
    max_epochs=9000,
    module__d_out=4,
    module__nonlin=nn.ReLU(),
    criterion=CELoss,
    callbacks=get_callbacks(patience=10),

    # should be set based on input shape
    module__d_in=None,

    # control params
    warm_start=False,
    verbose=10,
    # device='cpu',
    device='cuda',
    train_split=CVSplit(5),
    iterator_train__shuffle=True,
)

ae_kargs = dict(
    criterion=nn.MSELoss,
    lr=5e-4,
    max_epochs=50000,
    batch_size=128,
    train_split=CVSplit(5),
    predict_nonlinearity=None,
    warm_start=False,
    verbose=10,
    optimizer=torch.optim.Adam,
    # device='cpu',
    device='cuda',
    iterator_train__shuffle=True,
    module__hlayer_sizes=(128, 64),
    callbacks=[callbacks.EarlyStopping(patience=30, monitor="valid_loss", lower_is_better=True)],
    module__d_in=None,
)


class AeNet(NeuralNet):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred
        return super().get_loss(decoded, y_true, *args, **kwargs)
