import numpy as np
from math import floor, ceil

import torch
from torch import nn
import torch.nn.functional as F

import utils.loggers as lg


class Residual_CNN(nn.Module):
    def __init__(self, learning_rate, input_dim,  output_dim, hidden_layers, device):
        super().__init__()
        self._device = device
        self._learning_rate = learning_rate
        self._input_dim = list(input_dim)
        self._output_dim = output_dim
        self._hidden_layers = hidden_layers
        self._num_layers = len(hidden_layers)
        self._activation = nn.LeakyReLU(0.3)
        self._conv_layer = self._construct_conv2d(self._input_dim, hidden_layers[0]['filters'], hidden_layers[0]['kernel_size'])
        self._conv_layer_normalization = nn.BatchNorm2d(hidden_layers[0]['filters'])
        self._residual_conv_layer_1 = []
        self._residual_conv_layer_2 = []
        self._residual_normalization = []
        if self._num_layers > 1:
            for i, h in enumerate(hidden_layers[1:]):
                self._input_dim[-3] = hidden_layers[i - 1]['filters']
                self._residual_conv_layer_1.append(self._construct_conv2d(self._input_dim, h['filters'], h['kernel_size']).to(device))
                self._input_dim[-3] = h['filters']
                self._residual_conv_layer_2.append(self._construct_conv2d(self._input_dim, h['filters'], h['kernel_size']).to(device))
                self._residual_normalization.append(nn.BatchNorm2d(h['filters']).to(device))
        self._input_dim[-3] = hidden_layers[-1]['filters']
        flatten_length = self._input_dim[-2] * self._input_dim[-1]
        self._vh_conv_layer = self._construct_conv2d(self._input_dim, 1, (1, 1))
        self._vh_normalization = nn.BatchNorm2d(1)
        self._vh_dense_1 = nn.Linear(flatten_length, 20, bias=False)
        self._vh_dense_2 = nn.Linear(20, 1, bias=False)
        self._ph_conv_layer = self._construct_conv2d(self._input_dim, 2, (1, 1))
        self._ph_normalization = nn.BatchNorm2d(2)
        self._ph_dense = nn.Linear(2 * flatten_length, output_dim, bias=False)
        self._input_dim = input_dim
    
    def _calc_padding_margin(self, kernel_size):
        same_pad_u = floor((kernel_size[-2] - 1) / 2.0)
        same_pad_b = ceil((kernel_size[-2] - 1) / 2.0)
        same_pad_l = floor((kernel_size[-1] - 1) / 2.0)
        same_pad_r = ceil((kernel_size[-1] - 1) / 2.0)
        return (same_pad_l, same_pad_r, same_pad_u, same_pad_b)

    def _construct_conv2d(self, x, filters, kernel_size):
        func = nn.Conv2d(
            in_channels=x[-3],
            out_channels=filters,
            kernel_size=kernel_size,
            bias=False
        )
        return func
    
    def forward(self, x, y=None):
        bs = x.shape[0]
        x = torch.from_numpy(x)
        x = x.to(self._device)
        
        x = F.pad(x, self._calc_padding_margin(self._hidden_layers[0]['kernel_size']))
        x = self._conv_layer(x)
        x = self._conv_layer_normalization(x)
        x = self._activation(x)
        
        for i in range(self._num_layers-1):
            input_block = x
            x = F.pad(x, self._calc_padding_margin(self._hidden_layers[i]['kernel_size']))
            x = self._residual_conv_layer_1[i](x)
            x = self._residual_normalization[i](x)
            x = self._activation(x)
            x = F.pad(x, self._calc_padding_margin(self._hidden_layers[i]['kernel_size']))
            x = self._residual_conv_layer_2[i](x)
            x = self._residual_normalization[i](x)
            x = input_block + x
            x = self._activation(x)

        vh = self._vh_conv_layer(x)
        vh = self._vh_normalization(vh)
        vh = self._activation(vh)
        vh = vh.view(bs, -1)
        vh = self._vh_dense_1(vh)
        vh = self._activation(vh)
        vh = self._vh_dense_2(vh)
        vh = torch.tanh(vh)

        ph = self._ph_conv_layer(x)
        ph = self._ph_normalization(ph)
        ph = self._activation(ph)
        ph = ph.view(bs, -1)
        ph = self._ph_dense(ph)

        # model = Model(inputs=[main_input], outputs=[vh, ph])
        # model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
        #     optimizer=SGD(lr=self.learning_rate, momentum = config.MOMENTUM),
        #     loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        #     )
        if y is None:
            return vh, ph
        
        v_loss_func = nn.MSELoss(reduction='none')
        p_loss_func = nn.CrossEntropyLoss(reduction='none')
        v_loss = v_loss_func(vh.view(-1), torch.from_numpy(y['value_head']).to(self._device))
        p_loss = p_loss_func(ph, torch.argmax(torch.from_numpy(y['policy_head']).to(self._device), dim=1))
        
        history = {}
        history['loss'] = 0.5 * v_loss + 0.5 * p_loss
        history['value_head_loss'] = v_loss
        history['policy_head_loss'] = p_loss

        return history

    def convertToModelInput(self, state):
        inputToModel = state.binary #np.append(state.binary, [(state.playerTurn + 1)/2] * self.input_dim[1] * self.input_dim[2])
        inputToModel = np.reshape(inputToModel, self._input_dim)
        return inputToModel
