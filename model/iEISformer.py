import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer,ChannelEmbedding,ARDecoderLayer,ARDecoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = ChannelEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout)

        # Encoder
        enc_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ]
        self.encoder = Encoder(enc_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))

        # Decoder
        self.dec_embedding = nn.Linear(configs.dec_in, configs.d_model)
        dec_layers = [
            ARDecoderLayer(
                AttentionLayer(
                    FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            )
            for l in range(configs.d_layers)
        ]
        self.decoder = ARDecoder(dec_layers, norm_layer=torch.nn.LayerNorm(configs.d_model))

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_inp = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[-1]]).float().to(x_enc.device)
        dec_inp = torch.cat([x_dec[:,:self.label_len,:], dec_inp], dim=1).float()

        dec_out = self.dec_embedding(dec_inp)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, enc_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:,-self.pred_len:,:]