import torch
import torch.nn as nn
from transformer.Layers import EncoderLayer, DecoderLayer


class Model(nn.Module):
    def __init__(self, d_model=128, d_inner=64, n_head=6, d_k=128, d_v=128):
        super(Model, self).__init__()

        self.encoder = EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v,
                                    atten_mode="softmax")
        self.decoder = DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v,
                                    atten_mode="sinkhorn")
        self.pos_embedding_layer = nn.Linear(2, d_model)

    def forward(self, descs0, descs1, pts0, pts1, atten_mask, key_mask):
        descs0, descs1 = descs0.transpose(0, 1), descs1.transpose(0, 1)
        pts0, pts1 = pts0.transpose(0, 1), pts1.transpose(0, 1)

        if torch.cuda.is_available():
            descs0, descs1, pts0, pts1, atten_mask, key_mask = descs0.cuda(), descs1.cuda(), pts0.cuda(), pts1.cuda(), atten_mask.cuda(), key_mask.cuda()

        pos0 = self.pos_embedding_layer(pts0)
        pos1 = self.pos_embedding_layer(pts1)

        for _ in range(3):
            descs0 = descs0 + pos0
            descs1 = descs1 + pos1
            descs0 = self.encoder(enc_input=descs0, slf_attn_mask=atten_mask)[0]
            descs1 = self.encoder(enc_input=descs1, slf_attn_mask=atten_mask)[0]

            descs0_ = descs0 + pos0
            descs1_ = descs1 + pos1
            descs0, cross_AB = self.decoder(dec_input=descs0_, enc_output=descs1_, slf_attn_mask=atten_mask,
                                            dec_enc_attn_mask=atten_mask)
            descs1, cross_BA = self.decoder(dec_input=descs1_, enc_output=descs0_, slf_attn_mask=atten_mask,
                                            dec_enc_attn_mask=atten_mask)

        match_matrix = (torch.sum(cross_AB, dim=1) + torch.sum(cross_BA, dim=1).permute(0, 2, 1)) * 0.5

        return match_matrix
