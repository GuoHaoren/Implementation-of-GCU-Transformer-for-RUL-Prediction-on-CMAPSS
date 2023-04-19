import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import pdb
import math
from transformer.Layers import Encoder

class Gating(nn.Module):
    def __init__(self, m,d_model, seq_len): # 128,128
        super().__init__()
        self.m = m
        self.seq_len = seq_len
        # the reset gate r_i
        self.W_r = nn.Parameter(torch.Tensor(m, m))
        self.V_r = nn.Parameter(torch.Tensor(m, m))
        self.b_r = nn.Parameter(torch.Tensor(m))

        # the update gate u_i
        self.W_u = nn.Parameter(torch.Tensor(m, m))
        self.V_u = nn.Parameter(torch.Tensor(m, m))
        self.b_u = nn.Parameter(torch.Tensor(m))

        # the output
        self.W_e = nn.Parameter(torch.Tensor(m, d_model))
        self.b_e = nn.Parameter(torch.Tensor(d_model))

        self.init_weights()

        self.cnn_layers = nn.Sequential(
            nn.Conv1d(seq_len,seq_len, kernel_size=3, stride=1,padding=1),
        )
        

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.m)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):

        x = x
        x_i = x #only applying the gating on the current row even with the stack of 3 rows cames as input (1,1,30,14)
        h_i = self.cnn_layers(x) # shape becomes 1,1,1,14 as the nn.conv2d has output channel as 1 but the convolution is applied on whole past input (stack of three)
        r_i = torch.sigmoid(torch.matmul(h_i, self.W_r) + torch.matmul(x_i, self.V_r) + self.b_r)
        u_i = torch.sigmoid(torch.matmul(h_i, self.W_u) + torch.matmul(x_i, self.V_u) + self.b_u)
        # the output of the gating mechanism
        hh_i = torch.mul(h_i, u_i) + torch.mul(x_i, r_i)

        return torch.matmul(hh_i, self.W_e) + self.b_e # (the final output is 1,1,1,128 as the encoder has size of 128.)
        
class FixedPositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """


        #x = x + self.pe[:x.size(0), :]
        x = x.permute(2,0,1)
     
        x = x + self.pe[:x.size(0),:]
        x = self.layer_norm(self.dropout(x))
        return x.transpose(0,1)


class GCU_Transformer(nn.Module):
    def __init__(self, seq_size=50, patch_size=10, in_chans=14,
                 embed_dim=128, depth=2, num_heads=4,
                 decoder_embed_dim=50, decoder_depth=8, decoder_num_heads=4,
                 norm_layer=nn.LayerNorm, batch_size = 20):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        #self.patch_embed = PatchEmbedRUL(seq_size, patch_size, in_chans, embed_dim)#50, 10, 14, 128
        self.patch_size = patch_size
        self.seq_size = seq_size
        self.embed_dim = embed_dim
        self.input_size = in_chans

        self.seq_tf = nn.Sequential(
            nn.Linear(in_features=seq_size*embed_dim, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1),
        )

        self.gating = Gating(in_chans, embed_dim, seq_size)
        #self.PositionalEncoder = PositionalEncoder(embed_dim)
        self.RULPositionalEncoder = FixedPositionalEncoding(embed_dim, max_len=1024)

        self.encoder_block = nn.Sequential(Encoder(depth, num_heads,embed_dim//num_heads,embed_dim//num_heads,embed_dim,embed_dim//2),
                                            nn.Linear(embed_dim,embed_dim),)
                                           
        self.output1 = nn.Sequential(nn.Linear(in_features=10, out_features=1))
        self.initialize_weights()

    def initialize_weights(self):
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        #torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        x = self.RULPositionalEncoder(x.transpose(1,2)) 
        enc_x = self.encoder_block(x)
        return enc_x

    def forward(self, seq):
            gcu_out = self.gating(seq)
            latent_x = self.forward_encoder(gcu_out)
            y = latent_x.reshape(-1, self.seq_size * self.embed_dim)
            x_tf = self.seq_tf(y)
            out = self.output1(x_tf)

            return out
