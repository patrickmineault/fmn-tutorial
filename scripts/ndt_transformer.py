"""
A reimplementation of the Neural Data Transformer (NDT) for spike data.
This is adapted from the original codebase, modified to fit into a single file.
Original author: Joel Ye.
Original code: https://github.com/snel-repo/neural-data-transformers/tree/master
"""
#!/usr/bin/env python3
# Author: Joel Ye
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Poisson

def binary_mask_to_attn_mask(x):
    return x.float().masked_fill(x == 0, float('-inf')).masked_fill(x == 1, float(0.0))

class PositionalEncoding(nn.Module):
    r"""
    ! FYI - needs even d_model if not learned.
    """
    def __init__(self, cfg, trial_length, d_model, device):
        super().__init__()
        self.dropout = nn.Dropout(p=cfg.DROPOUT_EMBEDDING)
        pe = torch.zeros(trial_length, d_model).to(device) # * Can optim to empty
        position = torch.arange(0, trial_length, dtype=torch.float).unsqueeze(1)
        self.learnable = cfg.LEARNABLE_POSITION
        if self.learnable:
            self.register_buffer('pe', position.long())
            self.pos_embedding = nn.Embedding(trial_length, d_model) # So maybe it's here...?
        else:
            if cfg.POSITION.OFFSET:
                position = position + 1
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1) # t x 1 x d
            self.register_buffer('pe', pe)

    def update_config(self, config):
        self.dropout = nn.Dropout(config.DROPOUT_EMBEDDING)

    def forward(self, x):
        if self.learnable:
            x = x + self.pos_embedding(self.pe) # t x 1 x d
        else:
            x = x + self.pe[:x.size(0), :] # t x 1 x d, # t x b x d
        return self.dropout(x)


class TransformerEncoderLayerWithHooks(TransformerEncoderLayer):
    def __init__(self, config, d_model, device=None, **kwargs):
        super().__init__(
            d_model,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.HIDDEN_SIZE,
            dropout=config.DROPOUT,
            activation=config.ACTIVATION,
            **kwargs
            )
        self.config = config
        self.num_input = d_model
        self.device = device
        if config.FIXUP_INIT:
            self.fixup_initialization()

    def fixup_initialization(self):
        r"""
        http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        temp_state_dic = {}
        en_layers = self.config.NUM_LAYERS

        for name, param in self.named_parameters():
            if name in ["linear1.weight",
                        "linear2.weight",
                        "self_attn.out_proj.weight",
                        ]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param
            elif name in ["self_attn.v_proj.weight",]:
                temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * (param * (2**0.5))

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)


    def update_config(self, config):
        self.config = config
        self.dropout = nn.Dropout(config.DROPOUT)
        self.dropout1 = nn.Dropout(config.DROPOUT)
        self.dropout2 = nn.Dropout(config.DROPOUT)

    def get_input_size(self):
        return self.num_input

    def attend(self, src, context_mask=None, **kwargs):
        attn_res = self.self_attn(src, src, src, attn_mask=context_mask, **kwargs)
        return (*attn_res, torch.tensor(0, device=src.device, dtype=torch.float))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            src: L, N, E (time x batch x neurons)
            weights: N, L, S (batch x target time x source time)
        """
        residual = src
        if self.config.PRE_NORM:
            src = self.norm1(src)

        src2, weights, attention_cost = self.attend(
            src,
            context_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = residual + self.dropout1(src2)
        if not self.config.PRE_NORM:
            src = self.norm1(src)
        residual = src
        if self.config.PRE_NORM:
            src = self.norm2(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.config.PRE_NORM:
            src = self.norm2(src)

        return src, weights, attention_cost

class TransformerEncoderWithHooks(TransformerEncoder):
    r""" Hooks into transformer encoder.
    """
    def __init__(self, encoder_layer, norm=None, config=None, num_layers=None, device=None):
        super().__init__(encoder_layer, config.NUM_LAYERS, norm)
        self.device = device
        self.update_config(config)

    def update_config(self, config):
        self.config = config
        for layer in self.layers:
            layer.update_config(config)

    def split_src(self, src):
        r""" More useful in inherited classes """
        return src

    def extract_return_src(self, src):
        r""" More useful in inherited classes """
        return src

    def forward(self, src, mask=None, return_outputs=False, return_weights=False, **kwargs):
        src = self.split_src(src)
        for i, mod in enumerate(self.layers):
            src, weights, layer_cost = mod(src, src_mask=mask, **kwargs)

        return_src = self.extract_return_src(src)
        if self.norm is not None:
            return_src = self.norm(return_src)

        return return_src

class NeuralDataTransformer(nn.Module):
    r"""
        Transformer encoder-based dynamical systems decoder. Trained on MLM loss. Returns loss and predicted rates.
    """

    def __init__(self, config, trial_length, num_neurons, device, max_spikes):
        super().__init__()
        self.config = config
        self.trial_length = trial_length
        self.num_neurons = num_neurons
        self.device = device

        # TODO buffer
        if config.FULL_CONTEXT:
            self.src_mask = None
        else:
            self.src_mask = {} # multi-GPU masks

        if config.EMBED_DIM == 0:
            self.num_input = num_neurons
        else:
            self.num_input = config.EMBED_DIM * num_neurons

        if config.LINEAR_EMBEDDER:
            self.embedder = nn.Linear(self.num_neurons, self.num_input)
        elif config.EMBED_DIM == 0:
            self.embedder = nn.Identity()
        else:
            self.embedder = nn.Sequential(
                nn.Embedding(max_spikes + 2, config.EMBED_DIM),
                nn.Flatten(start_dim=-2)
            ) # t x b x n -> t x b x (h/num_input = n x self.embed_dim)

        self.scale = math.sqrt(self.num_input)
        self.pos_encoder = PositionalEncoding(config, trial_length, self.num_input, device)
        self._init_transformer()

        self.rate_dropout = nn.Dropout(config.DROPOUT_RATES)

        decoder_layers = []
        decoder_layers.append(nn.Linear(self.get_factor_size(), num_neurons))
        
        self.decoder = nn.Sequential(*decoder_layers)

        self.init_weights()

    def update_config(self, config):
        r"""
            Update config -- currently, just replaces config and replaces the dropout layers
        """
        self.config = config
        self.rate_dropout = nn.Dropout(config.DROPOUT_RATES)
        self.pos_encoder.update_config(config)
        self.transformer_encoder.update_config(config)
        self.src_mask = {} # Clear cache

    def get_factor_size(self):
        return self.num_input

    def get_hidden_size(self):
        return self.num_input

    def get_encoder_layer(self):
        return TransformerEncoderLayerWithHooks

    def get_encoder(self):
        return TransformerEncoderWithHooks

    def _init_transformer(self):
        assert issubclass(self.get_encoder_layer(), TransformerEncoderLayerWithHooks)
        assert issubclass(self.get_encoder(), TransformerEncoderWithHooks)
        encoder_layer = self.get_encoder_layer()(self.config, d_model=self.num_input, device=self.device)
        norm = nn.LayerNorm(self.get_factor_size())
        self.transformer_encoder = self.get_encoder()(
            encoder_layer,
            norm=norm,
            config=self.config,
            device=self.device
        )

    def _get_or_generate_context_mask(self, src, do_convert=True, expose_ic=True):
        if self.config.FULL_CONTEXT:
            return None
        if str(src.device) in self.src_mask:
            return self.src_mask[str(src.device)]
        size = src.size(0) # T
        context_forward = self.config.CONTEXT_FORWARD
        if self.config.CONTEXT_FORWARD < 0:
            context_forward = size
        mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_forward) == 1).transpose(0, 1)
        if self.config.CONTEXT_BACKWARD > 0:
            back_mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-self.config.CONTEXT_BACKWARD) == 1)
            mask = mask & back_mask
        if expose_ic and self.config.CONTEXT_WRAP_INITIAL and self.config.CONTEXT_BACKWARD > 0:
            # Expose initial segment for IC
            initial_mask = torch.triu(torch.ones(self.config.CONTEXT_BACKWARD, self.config.CONTEXT_BACKWARD, device=src.device))
            mask[:self.config.CONTEXT_BACKWARD, :self.config.CONTEXT_BACKWARD] |= initial_mask
        mask = mask.float()
        if do_convert:
            mask = binary_mask_to_attn_mask(mask)
        self.src_mask[str(src.device)] = mask
        return self.src_mask[str(src.device)]

    def init_weights(self):
        r"""
            Init hoping for better optimization.
            Sources:
            Transformers without Tears https://arxiv.org/pdf/1910.05895.pdf
            T-Fixup http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        initrange = 0.1
        if self.config.EMBED_DIM != 0:
            if hasattr(self.config, "SPIKE_LOG_INIT") and self.config.SPIKE_LOG_INIT:
                # Use a log scale, since we expect spike semantics to follow compressive distribution
                max_spikes = self.embedder[0].num_embeddings + 1
                log_scale = torch.arange(1, max_spikes).float().log() # 1 to lg
                log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                log_scale = log_scale * initrange
                # Add some noise
                self.embedder[0].weight.data.uniform_(-initrange / 10, initrange / 10)
                self.embedder[0].weight.data += log_scale.unsqueeze(1).expand_as(self.embedder[0].weight.data)
            else:
                self.embedder[0].weight.data.uniform_(-initrange, initrange)

        self.decoder[0].bias.data.zero_()
        self.decoder[0].weight.data.uniform_(-initrange, initrange)

    def forward(self, src, **kwargs):
        src = src.permute(1, 0, 2) # t x b x n
        src = self.embedder(src) * self.scale
        src = self.pos_encoder(src)
        src_mask = self._get_or_generate_context_mask(src)
        output = self.transformer_encoder(src, src_mask, **kwargs)
        rate_output = self.rate_dropout(output)
        pred_rates = self.decoder(rate_output)

        return pred_rates.permute(1, 0, 2)

_config = {
    "TRIAL_LENGTH": 0,
    "CONTEXT_FORWARD": 10,
    "CONTEXT_BACKWARD": 10,
    "CONTEXT_WRAP_INITIAL": False,  
    "FULL_CONTEXT": False,
    "HIDDEN_SIZE": 128,
    "DROPOUT": 0.5,
    "DROPOUT_RATES": 0.5,  # Sanity check
    "DROPOUT_EMBEDDING": 0.5,
    "NUM_HEADS": 1,
    "NUM_LAYERS": 4,
    "ACTIVATION": "relu",
    "LINEAR_EMBEDDER": False,  
    "EMBED_DIM": 0,
    "LEARNABLE_POSITION": True,  # Maybe True?
    "MAX_SPIKE_COUNT": 20,
    "LOGRATE": True,
    "SPIKE_LOG_INIT": False,
    "FIXUP_INIT": True,
    "PRE_NORM": True,
    "SCALE_NORM": False,
    "LOSS": {
        "TYPE": "poisson",
        "TOPK": 0.1,
    },
    "POSITION": {
        "OFFSET": True,  # Whether to offset the position encoding by 1
    },
    "DECODER": {
        "LAYERS": 1,  # Number of layers in the decoder
    },
}

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

def instantiate_autoencoder(args, n_neurons, trial_len, device):
    _config['HIDDEN_SIZE'] = args.ffn_dim
    _config['DROPOUT'] = args.dropout
    _config['DROPOUT_RATES'] = args.dropout
    _config['DROPOUT_EMBEDDING'] = args.dropout
    _config['NUM_HEADS'] = args.num_heads
    _config['NUM_LAYERS'] = args.num_layers
    _config['CONTEXT_BACKWARD'] = args.context
    _config['CONTEXT_FORWARD'] = args.context
    _config['LEARNABLE_POSITION'] = args.pos_encoding == "learned"

    config = DictObj(_config)

    return NeuralDataTransformer(
        config=config,
        trial_length=trial_len,
        num_neurons=n_neurons,
        device=device,
        max_spikes=20,
    )