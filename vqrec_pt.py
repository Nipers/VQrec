import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender


class VQRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # VQRec args
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.pq_codes = dataset.pq_codes

        # load parameters info
        self.pq_code_embedding = nn.Embedding(self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
        
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        pq_code_seq = self.pq_codes[item_seq]
        pq_code_emb = self.pq_code_embedding(pq_code_seq).mean(dim=-2)