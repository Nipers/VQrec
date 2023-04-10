import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender


def log(t, eps = 1e-6):
    return torch.log(t + eps)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)


def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)


def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)


class VQRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # VQRec args
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.pq_codes = dataset.pq_codes
        self.temperature = config['temperature']
        self.index_assignment_flag = False
        self.sinkhorn_iter = config['sinkhorn_iter']
        self.fake_idx_ratio = config['fake_idx_ratio']

        self.train_stage = config['train_stage']
        assert self.train_stage in [
            'pretrain', 'inductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.pq_code_embedding = nn.Embedding(
            self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
        self.reassigned_code_embedding = None

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            raise NotImplementedError()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def code_projection(self):
        doubly_stochastic_matrix = gumbel_sinkhorn(torch.exp(self.trans_matrix), n_iters=self.sinkhorn_iter)
        trans = differentiable_topk(doubly_stochastic_matrix.reshape(-1, self.code_cap + 1), 1)
        trans = torch.ceil(trans.reshape(-1, self.code_cap + 1, self.code_cap + 1))
        raw_embed = self.pq_code_embedding.weight.reshape(self.code_dim, self.code_cap + 1, -1)
        trans_embed = torch.bmm(trans, raw_embed).reshape(-1, self.hidden_size)
        return trans_embed
            
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        pq_code_seq = self.pq_codes[item_seq]
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(pq_code_seq, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding(pq_code_seq).mean(dim=-2)
        input_emb = pq_code_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_item_emb(self):
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(self.pq_codes, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb  # [B H]

    def generate_fake_neg_item_emb(self, item_index):
        rand_idx = torch.randint_like(input=item_index, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(item_index.device) * (self.code_cap + 1)).unsqueeze(0)
        rand_idx = rand_idx + base_id + 1
        
        mask = torch.bernoulli(torch.full_like(item_index, self.fake_idx_ratio, dtype=torch.float))
        fake_item_idx = torch.where(mask > 0, rand_idx, item_index)
        fake_item_idx[0,:] = 0
        return self.pq_code_embedding(fake_item_idx).mean(dim=-2)

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_id = interaction['item_id']
        pos_pq_code = self.pq_codes[pos_id]
        if self.index_assignment_flag:
            pos_items_emb = F.embedding(pos_pq_code, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pos_items_emb = self.pq_code_embedding(pos_pq_code).mean(dim=-2)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        fake_item_emb = self.generate_fake_neg_item_emb(pos_pq_code)
        fake_item_emb = F.normalize(fake_item_emb, dim=-1)
        fake_logits = (seq_output * fake_item_emb).sum(dim=1, keepdim=True) / self.temperature
        fake_logits = torch.exp(fake_logits)

        loss = -torch.log(pos_logits / (neg_logits + fake_logits))
        return loss.mean()
    
    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        return self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
    
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            raise NotImplementedError()
        else:  # self.loss_type = 'CE'
            test_item_emb = self.calculate_item_emb()
            
            if self.temperature > 0:
                seq_output = F.normalize(seq_output, dim=-1)
                test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            
            if self.temperature > 0:
                logits /= self.temperature
            
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        raise NotImplementedError()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.calculate_item_emb()
        
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_items_emb = F.normalize(test_items_emb, dim=-1)
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

class MMVQRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # VQRec args
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.temperature = config['temperature']
        self.index_assignment_flag = False
        self.sinkhorn_iter = config['sinkhorn_iter']
        self.fake_idx_ratio = config['fake_idx_ratio']
        self.topk = bool(config["top_k"])
        self.use_item_encoder = bool(config["item_encoder"])
        self.use_id_embd = bool(config["use_id_embd"])
        print(f"top_k: {self.topk}")
        print(f"use_id_embd: {self.use_id_embd}")

        self.train_stage = config['train_stage']
        assert self.train_stage in [
            'pretrain', 'inductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.item_num = dataset.item_num
        self.user_num = dataset.user_num

        # define layers and loss

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        # self.item_encoder = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.GELU()
        # )
        # self.gated_weight = nn.Sequential(
        #     nn.Linear(self.hidden_size * self.code_dim, self.code_dim),
        #     nn.GELU()
        # )
        # self.item_encoder = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.GELU()
        # )
        if self.use_item_encoder:
            self.item_encoder = TransformerEncoder(
                n_layers=1,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps
            )
            self.item_position_embedding = nn.Embedding(self.code_dim * 2, self.hidden_size)
        if self.use_id_embd:
            # without transformer, pure add
            self.item_id_embedding = nn.Embedding(self.item_num, 300)
            # self.user_encoder = nn.Sequential(
            #     nn.Linear(self.hidden_size, self.hidden_size),
            #     nn.GELU(),
            #     nn.Linear(self.hidden_size, self.hidden_size),
            #     nn.GELU(),
            # )
            # self.item_encoder = nn.Sequential(
            #     nn.Linear(self.hidden_size, self.hidden_size),
            #     nn.GELU(),
            #     nn.Linear(self.hidden_size, self.hidden_size),
            #     nn.GELU(),
            # )
            # self.id_embedding = nn.Embedding(self.item_num, self.hidden_size)
            # self.item_encoder = TransformerEncoder(
            #     n_layers=1,
            #     n_heads=self.n_heads,
            #     hidden_size=self.hidden_size,
            #     inner_size=self.inner_size,
            #     hidden_dropout_prob=self.hidden_dropout_prob,
            #     attn_dropout_prob=self.attn_dropout_prob,
            #     hidden_act=self.hidden_act,
            #     layer_norm_eps=self.layer_norm_eps
            # )
            # self.item_position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        if self.topk:
            self.pq_codes = dataset.pq_codes
            self.pq_code_embedding = nn.Embedding(
                self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
            # self.pq_code_embedding = nn.Embedding(
            #     self.code_dim * (1 + self.code_cap), int(self.hidden_size / 2), padding_idx=0)
            self.reassigned_code_embedding = None
            self.trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))
        else:        
            self.word_pq_codes = dataset.word_pq_codes
            self.image_pq_codes = dataset.image_pq_codes
            self.image_masks = dataset.image_masks
            self.word_pq_code_embedding = nn.Embedding(
                self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
            self.image_pq_code_embedding = nn.Embedding(
                self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
            self.word_reassigned_code_embedding = None
            self.image_reassigned_code_embedding = None
            self.word_trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))
            self.image_trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            raise NotImplementedError()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def code_projection(self):
        if self.topk:            
            doubly_stochastic_matrix = gumbel_sinkhorn(torch.exp(self.trans_matrix), n_iters=self.sinkhorn_iter)
            trans = differentiable_topk(doubly_stochastic_matrix.reshape(-1, self.code_cap + 1), 1)
            trans = torch.ceil(trans.reshape(-1, self.code_cap + 1, self.code_cap + 1))
            raw_embed = self.pq_code_embedding.weight.reshape(self.code_dim, self.code_cap + 1, -1)
            trans_embed = torch.bmm(trans, raw_embed).reshape(-1, self.hidden_size)
            # trans_embed = torch.bmm(trans, raw_embed).reshape(-1, int(self.hidden_size / 2))
            assert trans_embed.shape == self.pq_code_embedding.weight.shape
            return trans_embed
        else:
            word_doubly_stochastic_matrix = gumbel_sinkhorn(torch.exp(self.word_trans_matrix), n_iters=self.sinkhorn_iter)
            word_trans = differentiable_topk(word_doubly_stochastic_matrix.reshape(-1, self.code_cap + 1), 1)
            word_trans = torch.ceil(word_trans.reshape(-1, self.code_cap + 1, self.code_cap + 1))
            word_raw_embed = self.word_pq_code_embedding.weight.reshape(self.code_dim, self.code_cap + 1, -1)
            word_trans_embed = torch.bmm(word_trans, word_raw_embed).reshape(-1, self.hidden_size)

            image_doubly_stochastic_matrix = gumbel_sinkhorn(torch.exp(self.image_trans_matrix), n_iters=self.sinkhorn_iter)
            image_trans = differentiable_topk(image_doubly_stochastic_matrix.reshape(-1, self.code_cap + 1), 1)
            image_trans = torch.ceil(image_trans.reshape(-1, self.code_cap + 1, self.code_cap + 1))
            image_raw_embed = self.image_pq_code_embedding.weight.reshape(self.code_dim, self.code_cap + 1, -1)
            image_trans_embed = torch.bmm(image_trans, image_raw_embed).reshape(-1, self.hidden_size)
            return word_trans_embed, image_trans_embed

    def mm_pq_code(self, item_id):
        if self.topk:
            pq_code = self.pq_codes[item_id]
            return pq_code
        else:
            word_pq_code = self.word_pq_codes[item_id]
            image_pq_code = self.image_pq_codes[item_id]
            image_masks = self.image_masks[item_id]
            return word_pq_code, image_pq_code, image_masks
        
    def item_encode(self, word_pq_code, image_pq_code, image_masks):
        pq_code = torch.cat((word_pq_code, image_pq_code), dim=-1)
        assert pq_code.shape[-1] == self.code_dim * 2,f'pq_code: {pq_code.shape}, self.code_dim: {self.code_dim}'
        position_ids = torch.arange(pq_code.size(1), dtype=torch.long, device=pq_code.device)
        position_ids = position_ids.unsqueeze(0).expand_as(pq_code)
        position_embedding = self.item_position_embedding(position_ids)
        if self.index_assignment_flag:  
            items_emb = F.embedding(word_pq_code, self.word_reassigned_code_embedding, padding_idx=0)
            image_emb = F.embedding(image_pq_code, self.image_reassigned_code_embedding, padding_idx=0)             
        else:
            items_emb = self.word_pq_code_embedding(word_pq_code)
            image_emb = self.image_pq_code_embedding(image_pq_code)
        items_emb = torch.cat((items_emb, image_emb), dim=-2)
        assert items_emb.shape[-2] == self.code_dim * 2
        # item_num, 64, 300
        attention_mask = (pq_code != 0)[:,32:]
        image_masks = image_masks.unsqueeze(-1).expand_as(attention_mask)
        assert (attention_mask == image_masks).any()

        extended_attention_mask = self.get_attention_mask(pq_code)
        input_emb = items_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        items_emb = self.item_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)[0]
        items_emb = items_emb.mean(-2)     
        return items_emb   
        
    
    def mm_item_emb(self, word_pq_code, image_pq_code, image_masks, item_seq=None):
        items_emb = self.word_pq_code_embedding(word_pq_code)
        image_items_emb = self.image_pq_code_embedding(image_pq_code)
        items_emb[image_masks] += image_items_emb[image_masks]
        items_emb[image_masks] /= 2
        if self.use_id_embd:
            id_embd = self.item_id_embedding(item_seq).unsqueeze(-2)
            items_emb = torch.cat((items_emb, id_embd), dim=-2)
        items_emb = items_emb.mean(dim=-2)
        return items_emb
        # items_emb = self.word_pq_code_embedding(word_pq_code).mean(dim=-2)
        # image_items_emb = self.image_pq_code_embedding(image_pq_code).mean(dim=-2)
        # items_emb[image_masks] += image_items_emb[image_masks]
        # items_emb[image_masks] /= 2            
        # return items_emb
    
    def mm_item_emb_topk(self, pq_code):
        items_emb = self.pq_code_embedding(pq_code)
        items_emb = items_emb.mean(dim=-2)
        # print(pq_code.shape)
        # items_emb =  items_emb.reshape(list(items_emb.shape[:-2]) + [-1]).mean(dim=-2)
        # print(items_emb.shape)
        return items_emb
    
    def reassigned_mm_item_emb(self, word_pq_code_seq, image_pq_code_seq, image_masks):
        items_emb = F.embedding(word_pq_code_seq, self.word_reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        image_items_emb = F.embedding(image_pq_code_seq, self.image_reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        items_emb[image_masks] += image_items_emb[image_masks]
        items_emb[image_masks] /= 2
        return items_emb               
    
    def reassigned_mm_item_embd_topk(self, pq_code_seq):
        items_emb = F.embedding(pq_code_seq, self.reassigned_code_embedding, padding_idx=0)
        # items_emb = items_emb.reshape(list(items_emb.shape[:-2]) + [-1]).mean(dim=-2)
        return items_emb.mean(dim=-2)
    
    def generate_fake_neg_item_emb(self, word_pos_pq_code, image_pos_pq_code, image_masks):
        word_rand_idx = torch.randint_like(input=word_pos_pq_code, high=self.code_cap)
        image_rand_idx = torch.randint_like(input=image_pos_pq_code, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(word_pos_pq_code.device) * (self.code_cap + 1)).unsqueeze(0)
        word_rand_idx = word_rand_idx + base_id + 1
        image_rand_idx = image_rand_idx + base_id + 1
        
        word_mask = torch.bernoulli(torch.full_like(word_pos_pq_code, self.fake_idx_ratio, dtype=torch.float))
        image_mask = torch.bernoulli(torch.full_like(image_pos_pq_code, self.fake_idx_ratio, dtype=torch.float))
        fake_word_pq_code = torch.where(word_mask > 0, word_rand_idx, word_pos_pq_code)
        # fake_word_pq_code[0,:] = 0
        fake_image_pq_code = torch.where(torch.logical_and(image_mask > 0, image_masks.unsqueeze(-1).expand_as(image_mask)), image_rand_idx, word_pos_pq_code)
        # fake_image_pq_code[0,:] = 0
        if self.use_item_encoder:
            fake_items_emb = self.item_encode(fake_word_pq_code, fake_image_pq_code, image_masks)
        else:
            fake_items_emb = self.mm_item_emb(fake_word_pq_code, fake_image_pq_code, image_masks)
        return fake_items_emb

    def generate_fake_neg_item_emb_topk(self, pos_pq_code):
        rand_idx = torch.randint_like(input=pos_pq_code, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(pos_pq_code.device) * (self.code_cap + 1)).unsqueeze(0)
        # base_id = (torch.arange(self.code_dim).to(pos_pq_code.device) * (self.code_cap + 1)).unsqueeze(0).unsqueeze(-1)
        rand_idx = rand_idx + base_id + 1
        
        mask = torch.bernoulli(torch.full_like(pos_pq_code, self.fake_idx_ratio, dtype=torch.float))
        fake_pq_code = torch.where(mask > 0, rand_idx, pos_pq_code)
        # fake_item_idx[0,:] = 0
        fake_item_embd = self.mm_item_emb_topk(fake_pq_code)
        return fake_item_embd
       
    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_id = interaction['item_id']
        if self.topk:
            pos_pq_code = self.pq_codes[pos_id]
            if self.index_assignment_flag:
                pos_items_emb = self.reassigned_mm_item_embd_topk(pos_pq_code)
            else:
                pos_items_emb = self.mm_item_emb_topk(pos_pq_code)
            fake_item_emb = self.generate_fake_neg_item_emb_topk(pos_pq_code)  
        else:
            word_pos_pq_code, image_pos_pq_code, image_masks = self.mm_pq_code(pos_id)
            if self.use_item_encoder:
                pos_items_emb = self.item_encode(word_pos_pq_code, image_pos_pq_code, image_masks)
            else:
                if self.index_assignment_flag:
                    pos_items_emb = F.embedding(word_pos_pq_code, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
                else:
                    pos_items_emb = self.mm_item_emb(word_pos_pq_code, image_pos_pq_code, image_masks)   
            fake_item_emb = self.generate_fake_neg_item_emb(word_pos_pq_code, image_pos_pq_code, image_masks)
        assert pos_items_emb.shape == fake_item_emb.shape, f"{pos_items_emb.shape}, {fake_item_emb.shape}"
        pos_items_emb = self.item_encoder(pos_items_emb)
        fake_item_emb = self.item_encoder(fake_item_emb)
        pos_items_emb = F.normalize(pos_items_emb, dim=-1)
        fake_item_emb = F.normalize(fake_item_emb, dim=-1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)
        # 可以把neg的定义改一下看看
        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        fake_logits = (seq_output * fake_item_emb).sum(dim=1, keepdim=True) / self.temperature
        fake_logits = torch.exp(fake_logits)

        loss = -torch.log(pos_logits / (neg_logits + fake_logits))
        return loss.mean()
    
    def forward(self, item_seq, item_seq_len, users=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_ids = (item_seq_len - 1).unsqueeze(-1).expand_as(item_seq) - position_ids 
        position_ids[position_ids < 0] = 0
        position_embedding = self.position_embedding(position_ids)
        if self.topk:
            pq_codes = self.mm_pq_code(item_seq)
            if self.index_assignment_flag:
                pq_code_emb = self.reassigned_mm_item_embd_topk(pq_codes)
            else:
                pq_code_emb = self.mm_item_emb_topk(pq_codes)
        else:
            if self.use_item_encoder:
                unique_ids = torch.unique(item_seq).to(item_seq.device)
                id2index = torch.zeros((int(unique_ids.max() + 1)), dtype=torch.long).to(item_seq.device)
                index = torch.arange(0, int(unique_ids.shape[0]), dtype=torch.long).to(item_seq.device)
                id2index[unique_ids[index]] = index
                item_seq_indice = id2index[item_seq]
                word_pq_code_seq, image_pq_code_seq, image_masks = self.mm_pq_code(unique_ids)
                items_embedding = self.item_encode(word_pq_code_seq, image_pq_code_seq, image_masks)
                pq_code_emb = F.embedding(item_seq_indice, items_embedding)
            else:
                word_pq_code_seq, image_pq_code_seq, image_masks = self.mm_pq_code(item_seq)
                if self.index_assignment_flag:
                    pq_code_emb = self.reassigned_mm_item_emb(word_pq_code_seq, image_pq_code_seq, image_masks)
                else:
                    pq_code_emb = self.mm_item_emb(word_pq_code_seq, image_pq_code_seq, image_masks, item_seq)
            # pq_code_emb = self.pq_code_embedding(pq_code_seq).mean(dim=-2)
        input_emb = pq_code_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)[0]
        output = self.gather_indexes(output, item_seq_len - 1)
        if self.use_id_embd:
            id_output = self.user_id_embedding(users)
            # id_embd = self.user_id_embedding(users)
            # id_output = self.user_encoder(id_embd)
            # item_pos_embd =  self.item_position_embedding(position_ids)
            # id_embd = self.id_embedding(item_seq)
            # id_embd += item_pos_embd
            # id_embd = self.LayerNorm(id_embd)
            # id_embd = self.dropout(id_embd)
            # id_output = self.item_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)[0]
            # id_output = self.gather_indexes(id_output, item_seq_len - 1)
            return output, id_output
        return output  # [B H]

    def calculate_item_emb(self):
        if self.topk:
            if self.index_assignment_flag:
                item_emb = self.reassigned_mm_item_embd_topk(self.pq_codes)
            else:
                item_emb = self.mm_item_emb_topk(self.pq_codes)
        else:
            if self.use_item_encoder:
                item_emb = self.item_encode(self.word_pq_codes, self.image_pq_codes, self.image_masks)
            else:
                item_seq = None
                if self.use_id_embd:
                    item_seq = torch.arange(start=0, end=self.item_num, dtype=torch.long, device=self.device)
                if self.index_assignment_flag:
                    item_emb = self.reassigned_mm_item_emb(self.word_pq_codes, self.image_pq_codes, self.image_masks)
                else:
                    item_emb = self.mm_item_emb(self.word_pq_codes, self.image_pq_codes, self.image_masks, item_seq)
        # if self.use_id_embd:
        #     id_embd = self.item_id_embedding.weight
        #     id_embd = self.item_encoder(id_embd)
        #     return item_emb, id_embd
        # if not self.word_trans_matrix.requires_grad:
        #     item_emb = self.item_encoder(item_emb)
        return item_emb  # [B H]

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        return self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
    
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        # print(interaction)
        users = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # if self.use_id_embd:
        #     seq_output, seq_id_output = self.forward(item_seq, item_seq_len, users)
        #     seq_output = torch.cat((seq_output, seq_id_output), dim=-1)
        # else:
        #     seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            raise NotImplementedError()
        else:  # self.loss_type = 'CE'
            test_item_emb = self.calculate_item_emb()
            
            if self.temperature > 0:
                seq_output = F.normalize(seq_output, dim=-1)
                test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            
            if self.temperature > 0:
                logits /= self.temperature
            
            loss = self.loss_fct(logits, pos_items)
            return loss
            # if self.use_id_embd:
            #     test_item_emb, test_item_id_emb = self.calculate_item_emb()
            #     test_item_emb = torch.cat((test_item_emb, test_item_id_emb), dim=-1)
            # else:                
            #     test_item_emb = self.calculate_item_emb()
            # if self.temperature > 0:
            #     seq_output = F.normalize(seq_output, dim=-1)
            #     test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            # logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # # if self.use_id_embd:
            # #     if self.temperature > 0:
            # #         seq_id_output = F.normalize(seq_id_output, dim=-1)
            # #         test_item_id_emb = F.normalize(test_item_id_emb, dim=-1)
            # #     logits_id = torch.matmul(seq_id_output, test_item_id_emb.transpose(0, 1))
            #     # if self.temperature > 0:
            #     #     logits_id /= self.temperature
            # if self.temperature > 0:
            #     logits /= self.temperature
            # if self.use_id_embd:
            #     # loss = self.loss_fct(logits, pos_items) + self.loss_fct(logits_id, pos_items)
            #     loss = self.loss_fct(logits, pos_items)
            #     # loss = self.loss_fct(logits + logits_id, pos_items)
            # else:
            #     loss = self.loss_fct(logits, pos_items)
            # return loss

    def predict(self, interaction):
        raise NotImplementedError()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.calculate_item_emb()
        
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_items_emb = F.normalize(test_items_emb, dim=-1)
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
        # users = interaction[self.USER_ID]
        # item_seq = interaction[self.ITEM_SEQ]
        # item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # if self.use_id_embd:
        #     seq_output, seq_id_output = self.forward(item_seq, item_seq_len, users)
        #     test_items_emb, test_items_id_emb = self.calculate_item_emb()   
        #     seq_output = torch.cat((seq_output, seq_id_output), dim=-1)      
        #     test_items_emb = torch.cat((test_items_emb, test_items_id_emb), dim=-1)    
        # else:
        #     seq_output = self.forward(item_seq, item_seq_len)
        #     test_items_emb = self.calculate_item_emb()
        
        # if self.temperature > 0:
        #     seq_output = F.normalize(seq_output, dim=-1)
        #     test_items_emb = F.normalize(test_items_emb, dim=-1)
        #     # if self.use_id_embd:
        #     #     seq_id_output = F.normalize(seq_id_output, dim=-1)
        #     #     test_items_id_emb = F.normalize(test_items_id_emb, dim=-1)
        #     #     scores_id = torch.matmul(seq_id_output, test_items_id_emb.transpose(0, 1))  # [B n_items]

        
        # scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        # # if self.use_id_embd:
        # #     scores += scores_id
        # # return scores_id
        # return scores 

class IDRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ID_rec args
        self.train_stage = config['train_stage']
        self.index_assignment_flag = False
        assert self.train_stage in [
            'pretrain', 'inductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'
        # self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.temperature = config['temperature']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        # define layers and loss

        # self.user_embedding = nn.Embedding(self.user_num, int(self.hidden_size), padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num, int(self.hidden_size), padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            raise NotImplementedError()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")
    
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_embedding = self.item_embedding(item_seq)
        input_emb = item_embedding + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)[0]
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]
    
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            raise NotImplementedError()
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            
            if self.temperature > 0:
                seq_output = F.normalize(seq_output, dim=-1)
                test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            
            if self.temperature > 0:
                logits /= self.temperature
            
            loss = self.loss_fct(logits, pos_items)
            return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_items_emb = F.normalize(test_items_emb, dim=-1)
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    