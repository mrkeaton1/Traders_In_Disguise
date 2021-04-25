import torch

from torch import nn

class TransformerClassifier(nn.Module):

    def __init__(self, d_model, n_heads, channels, n_hid, n_eclayers, dropout_p):
        super(TransformerClassifier, self).__init__()

        self.batch_norm_1 = nn.BatchNorm1d(channels)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, n_hid, dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, n_eclayers, norm=nn.LayerNorm(channels, d_model))

        self.to_logits = nn.Linear(d_model, 2)

        self.to_logits.weight.data.uniform_(-.01, .01)
        self.to_logits.bias.data.zero_()

    def forward(self, src, src_mask):

        # bn1 = self.batch_norm_1(src)
        te_out = self.transformer_encoder(src)
        # te_max = torch.max(te_out, dim=1)[0]
        te_max = te_out.max(dim=1)[0]
        # f_te = torch.flatten(te_max, 1, -1)
        result = self.to_logits(te_max)

        return result


if __name__ == '__main__':
    tc = TransformerClassifier(7, 7, 2048, 6, 0.5)
    src = torch.rand(4, 32, 7)

    out = tc(src, None)

    print('test')
