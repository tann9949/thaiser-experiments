from collections import OrderedDict
from typing import List, Dict, Any, Optional, Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..model.base_model import BaseModel


class SqueezeExcite(pl.LightningModule):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,  # context for pooling. If -1 = gap
        interpolation_mode: str = 'nearest',
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode

        self.pool = None  # prepare a placeholder which will be updated
        self.change_context_window(context_window=context_window)

        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def _se_pool_step(self, x):
        if self.context_window < 0:
            y = self.pool(x)
        else:
            y = _se_pool_step_script(x, self.context_window)
        return y

    def change_context_window(self, context_window: int):
        """
        Update the context window of the SqueezeExcitation module, in-place if possible.
        Will update the pooling layer to either nn.AdaptiveAvgPool1d() (for global SE) or nn.AvgPool1d()
        (for limited context SE).
        If only the context window is changing but still a limited SE context block - then
        the earlier instance of nn.AvgPool1d() will be updated.
        Args:
            context_window: An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.
                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
        """
        if hasattr(self, 'context_window'):
            logging.info(f"Changing Squeeze-Excitation context window from {self.context_window} to {context_window}")

        self.context_window = context_window

        if self.context_window < 0:
            if not isinstance(self.pool, nn.AdaptiveAvgPool1d):
                self.pool = nn.AdaptiveAvgPool1d(1)  # context window = T
        else:
            if not isinstance(self.pool, nn.AvgPool1d):
                self.pool = nn.AvgPool1d(self.context_window, stride=1)
            else:
                # update the context window
                self.pool.kernel_size = _single(self.context_window)
    
    def forward(self, x):
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        # Computes in float32 to avoid instabilities during training with AMP.
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            y = self._se_pool_step(x)
            y = y.transpose(1, -1)  # [B, T - context_window + 1, C]
            y = self.fc(y)  # [B, T - context_window + 1, C]
            y = y.transpose(1, -1)  # [B, C, T - context_window + 1]
        if self.context_window >= 0:
            y = F.interpolate(y, size=x.shape[-1], mode=self.interpolation_mode)

        y = torch.sigmoid(y)
        return x * y
    
    
def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation=1) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2
    
    
class JasperBlock(pl.LightningModule):
    def __init__(
        self,
        inplanes: int,  # in filters
        planes: int,  # out filters
        repeat: int = 3,
        kernel_size: List[int] = [11],
        kernel_size_factor: int = 1,
        stride: List[int] = [1],
        padding: str = "same",
        dropout: float = 0.2,
        activation: Optional[str] = None,
        residual: bool = True,
        groups: int = 1,
        separable: bool = False,
        heads: int = -1,
        normalization: str = "batch",
        norm_groups: int = 1,
        residual_mode: str = "add",
        residual_panes: List[int] = [],
        se: bool = False,
        se_reduction_ratio: int = 16,
        se_context_window: int = -1,
        se_interpolation_mode: str = "nearest",
        stride_last: bool = False
    ) -> None:
        super().__init__()
            
        # rescale kernel_size_factor (alpha in paper)
        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)
            
        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")
        padding_val = get_same_padding(kernel_size[0], stride[0])
            
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se
        
        inplanes_loop = inplanes
        conv = nn.ModuleList()
        
        # repeat sub blocks
        for _ in range(repeat - 1):
            stride_val = [1] if stride_last else stride
            
            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    padding=padding_val,
                    groups=groups,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                )
            )
            
            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            
        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_val,
                groups=groups,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups
            )
        )
        
        # squeeze excite
        if se:
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    activation=activation
                )
            )
        
        self.mconv = conv
        
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        
        # residual connection
        if residual:
            res_list = nn.ModuleList()
            stride_val = stride if residual_mode == "stride_add" else [1]
        
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val,
                    )
                )
                res_list.append(res)
            self.res = res_list
        else:
            self.res = None
        
        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))  
        
    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        separable=False
    ):
        return nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups
        )
    
    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        separable=False,
        normalization="batch",
        norm_groups=1
    ):
        if norm_groups == -1:
            norm_groups = out_channels
            
        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=bias,
                    groups=groups
                )
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            ]
            
        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match" f" one of [batch, layer, group, instance]."
            )
            
        return layers
    
    def _get_act_dropout_layer(
        self,
        drop_prob=0.2,
        activation=None
    ):
        if activation is None:
            activation = nn.Hardtanh(min_val=0., max_val=20.)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers
    
    def forward(
        self,
        x
    ):
        out = x
        for i, l in enumerate(self.mconv):
            out = l(out)
            
        # compute residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = x
                for j, res_layer in enumerate(layer):
                    res_out = res_layer(res_out)
                    
                if self.residual_mode == "add" or self.residual_mode == "stride_add":
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)
        
        # compute output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out]
        return out


class CitriMegaBlock(pl.LightningModule):
    def __init__(
        self,
        filters: int,
        kernels: List[int],
        dropout: float,
        se_reduction: int,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.kernels = kernels
        self.dropout = dropout
        self.se_reduction = se_reduction        
        
        # main block
        self.blocks = []
        
        k = self.kernels[0]
        self.blocks.append(
            (
                f"block-1",
                nn.Sequential(OrderedDict([
                    (f"conv", nn.Conv1d(self.filters, self.filters, k, 2, (k - 1) // 2)),
                ]))
            )
        )
        
        for i, k in enumerate(self.kernels[1:]):
            self.blocks.append(
                (
                    f"block-{i+2}",
                    nn.Sequential(OrderedDict([
                        (f"conv", nn.Conv1d(self.filters, self.filters, k, padding=(k - 1) // 2)),
                        (f"batchnorm", nn.BatchNorm1d(self.filters)),
                        (f"relu", nn.ReLU()),
                        (f"dropout", nn.Dropout(self.dropout))
                    ]))
                )
            )
        
        
            
        self.blocks = nn.Sequential(OrderedDict(self.blocks))
        
        # last block
        self.squeeze_excite = SqueezeExcite(filters, self.se_reduction)
        self.last_conv = nn.Sequential
        
        
    def forward(self, x):
        residual = self.residual_cnn(x)
        
        for cnn in self.blocks:
            x = cnn(x)
            
        print(x.shape, residual.shape)
        x = residual + x
        
        x = nn.ReLU()(x)
        x = nn.Dropout(self.dropout)(x)
            
        return x
    

class CitriNet(BaseModel):
    def __init__(
        self,
        hparams: Dict[str, Any] = None,
        schedule_learning_rate: bool = False,
        **kwargs
    ) -> None:
        if hparams is None:
            hparams = {}
        super().__init__(hparams, schedule_learning_rate, **kwargs)
        self.n_classes = hparams.get("n_classes", 4)
        self.in_channel = hparams.get("in_channel", 80)
        self.filters = hparams.get("filters", 256)
        self.dropout = hparams.get("dropout", 0.)
        self.se_reduction = hparams.get("se_reduction", 8)
        self.normalization = hparams.get("normalization", "batchnorm")
        self.blocks = hparams.get("blocks", {
            "mega-1": [5, 7, 7, 9, 9, 11],
            "mega-2": [7, 7, 9, 9, 11, 11, 13],
            "mega-3": [13, 13, 15, 15, 17, 17, 19, 19]
        })
        
        self.prolog_block = nn.Sequential(OrderedDict([
            ("prolog-conv", nn.Conv1d(
                in_channels=self.in_channel,
                out_channels=self.filters,
                kernel_size=5,
                padding=(5 - 1) // 2
            )),
            ("prolog-batchnorm", nn.BatchNorm1d(self.filters)),
            ("prolog-relu", nn.ReLU())
        ]))
        self.mega_blocks = nn.Sequential(OrderedDict([
            (name, CitriMegaBlock(self.filters, kernels, self.dropout, self.se_reduction)) 
            for name, kernels 
            in self.blocks.items()
        ]))
        self.epilog_block = nn.Sequential(OrderedDict([
            ("epilog-conv", nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=41,
                padding=(41 - 1) // 2
            )),
            ("epilog-batchnorm", nn.BatchNorm1d(self.filters)),
            ("epilog-relu", nn.ReLU())
        ]))
        self.logits = nn.Conv1d(self.filters, self.n_classes, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.prolog_block(x)
        x = self.mega_blocks(x)
        x = self.epilog_block(x)
        x = self.logits(x)
        x = x.mean(-1)
        return x
