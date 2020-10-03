import math
import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


class Model_Long(nn.Module):
    def __init__(self, config):
        super(Model_Long, self).__init__()
        self.emb_dim = config['emb_dim']

    def forward(self, u, v, t, scale, h_mask=None, return_w=False, flag='train'):
        u = u.unsqueeze(dim=2)  # n x d x 1  

        w = torch.matmul(v, u) * scale
        if flag == 'train':
            h_mask = h_mask.unsqueeze(dim=2)  
            w = w.masked_fill(h_mask == 0, -1e9)  
        w = torch.softmax(w, dim=1)  

        out = torch.sum(w.mul(v), dim=1)  

        if not return_w:
            pred_y = torch.sum((u.squeeze() + out).mul(t), dim=1)  # 1 x n
            pred_y = torch.sigmoid(pred_y)
            return pred_y
        else:
            # return w, out
            return w, (out + u.squeeze())


class MANN_Item(nn.Module):
    def __init__(self, config):
        super(MANN_Item, self).__init__()
        self.emb_dim = config['emb_dim']

    def forward(self, q, v, scale, h_mask=None, flag='train'):

        q = q.unsqueeze(dim=2)  # n x d x 1  

        w = torch.matmul(v, q) * scale
        if flag == 'train':
            h_mask = h_mask.unsqueeze(dim=2)  # n x m x 1
            w = w.masked_fill(h_mask == 0, -1e9)  
        w = torch.softmax(w, dim=1)

        p_m = torch.sum(w.mul(v), dim=1)  # n x d

        return p_m
        # return w, (out + u.squeeze())


class MANN_Feature(nn.Module):
    def __init__(self, config):
        super(MANN_Feature, self).__init__()
        self.config = config
        self.user_nums = config['user_nums']
        self.emb_dim = config['emb_dim']
        self.key_dim = config['key_dim']
        self.value_dim = config['value_dim']
        self.slot_num = config['slot_num']
        self.hop_num = config['hop_num']
        self.dropout_rate = config['dropout']
        bound = 1.0 / math.sqrt(self.emb_dim)

        self.Key = nn.Parameter(torch.randn(self.slot_num, self.key_dim))
        # self.Value = nn.Parameter(torch.randn(self.slot_num, self.value_dim))
        self.Value = nn.Parameter(torch.randn(self.user_nums, self.slot_num, self.key_dim))
        # self.Key = torch.nn.init.uniform_(self.Key, -bound, bound)
        # self.Value = torch.nn.init.uniform_(self.Value, -bound, bound)
        self.Key = torch.nn.init.normal_(self.Key, 0, 1.0)
        self.Value = torch.nn.init.normal_(self.Value, 0, 1.0)

        self.erase = nn.Linear(self.emb_dim, self.value_dim)
        self.add = nn.Linear(self.emb_dim, self.value_dim)

    def forward(self, user, query, s_mask=None, flag='train'):
        batch_len = user.shape[0]
        # user: batch_len * dim
        # query: batch_len * dim (B, D)

        # ==========shared=============
        # MK = self.Key  # slot_num * dim
        # MV = self.Value  # slot_num * dim
        # weight_qk = torch.softmax(torch.matmul(query, MK.t()), dim=-1)  # batch_len * slot_num
        # p_m = torch.matmul(weight_qk, MV)  # batch_len * dim

        # ==========privated============
        MK = self.Value[user]   # (B, K, D)
        weight_qk = torch.softmax(torch.matmul(MK, query.unsqueeze(2)), dim=1)  # (B, K, 1)
        p_m = torch.sum(weight_qk.mul(MK), dim=1)   # (B, D)

        # if flag == 'test':
        #     p_m = torch.matmul(weight_qk, MV)  # batch_len * dim
        #     # p_u = p_m
        #     # p_u = user + self.alpha * p_m
        # else:
        #     read = []
        #     for i in range(batch_len):
        #         # read
        #         temp = torch.matmul(weight_qk[i, :].unsqueeze(0), MV) #torch.sum(torch.mul(weight_qk[i,:], MV), dim=1)  # 1 * dim
        #         read.append(temp)
        #         # write
        #         erase_ratio = torch.sigmoid(self.erase(target[i]).unsqueeze(0))  # 1 * dim
        #         reverse_ratio = 1 - torch.einsum('bi,bj->bij',[weight_qk[i].unsqueeze(0), erase_ratio])  # 1 * slot_num * dim
        #         add_part = torch.tanh(self.add(query[i].unsqueeze(0)))  # 1 * dim
        #         add_part = torch.einsum('bi,bj->bij', [weight_qk[i].unsqueeze(0), add_part])  # 1 * slotNum * dim
        #         MV = MV * reverse_ratio.squeeze() + add_part.squeeze()
        #     p_m = torch.cat(read, dim=0)  # batch_len * dim
        #     # p_u = user + self.alpha * p_m

        return p_m


class Model_RNN(nn.Module):
    def __init__(self, config):
        super(Model_RNN, self).__init__()
        self.emb_dim = config['emb_dim']
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.rnn_type = config['rnn_type']
        self.num_drirections = config['num_directions']

        bidirectional = False
        if self.num_drirections == 2:
            bidirectional = True
        if self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=self.emb_dim, hidden_size=self.hidden_dim,
                                    num_layers=1, batch_first=True, dropout=0, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_dim,
                                    num_layers=1, batch_first=True, dropout=0, bidirectional=bidirectional)
        elif self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim,
                                    num_layers=1, batch_first=True, dropout=0, bidirectional=bidirectional)

        self.fc = torch.nn.Linear(self.hidden_dim, self.emb_dim)
    def forward(self, s, mask=None, flag='train'):
        # s: seq_items   t: targets
        # inputs are all embedding vectors
        batch_size = s.shape[0]
        h0 = torch.zeros(self.num_drirections, batch_size, self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_drirections, batch_size, self.hidden_dim).cuda()
        # if torch.cuda.is_available():
        #     h0.cuda()
        #     c0.cuda()
        if self.rnn_type == 'RNN' or self.rnn_type == 'GRU':
            out, hn = self.RNN(s, h0)           # if 双向：out--(batch_size, seq_len, num_directions * hidden_size)
                                                        #  hn--(num_directions, batch_size, hidden_size)
                                                # if 单项：out--(batch_size, seq_len, hidden_size)
        elif self.rnn_type == 'LSTM':
            out, (hn, cn) = self.RNN(s, (h0, c0))

        if self.num_drirections == 2:
            out = out.view(out.shape[0], out.shape[1], self.num_drirections, self.hidden_dim)  # (batch_size, seq_len, num_directions, hidden_size)
            out = torch.sum(out, dim=2)  #(batch_size, seq_len,hidden_size)
            hn = torch.sum(hn, dim=0)  # (batch_size, hidden_size)

        if mask is not None:
            out = mask.float().unsqueeze(-1).mul(out)

        return out, hn

# Transformer implemented by Pytorch
# class Model_Self_Attention(nn.Module):
#     def __init__(self, emb_dim, num_heads, seq_len, dim_feedforward=64, dropout=0.0):
#         super(Model_Self_Attention, self).__init__()
#         self.transformer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
#                                                             dim_feedforward=dim_feedforward, dropout=dropout)
#
#     def forward(self, x, mask=None, flag='train'):
#         x = x.transpose(0, 1)
#         if mask is not None:
#             out = self.transformer(x, src_key_padding_mask=(mask==0))
#         else:
#             out = self.transformer(x)
#         out = out.transpose(0, 1)
#         return out


class Model_Self_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, seq_len, dim_feedforward=64, dropout=0.0):
        super(Model_Self_Attention, self).__init__()
        self.emb_dim = emb_dim
        # self.positon_encode = self.positional_encoding(seq_len, emb_dim)
        self.position_encode = nn.Embedding(seq_len, emb_dim)
        self.position_index = torch.arange(0, seq_len, dtype=torch.long).cuda()
        self.attn = MultiheadAttention(emb_dim, num_heads)

        # Position-wise-Feed-Forward-Networks
        self.linear1 = nn.Linear(emb_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)

        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def positional_encoding(self, seq_len, emb_dim):
        position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / emb_dim) for j in range(emb_dim)] for pos in range(seq_len)])
        # apply sin to even indices in the array; 2i
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        return T.FloatTensor(position_encoding, requires_grad=False)

    def forward(self, x, dist, mask=None, flag='train'):
        if flag == 'test':
            self.position_index = torch.arange(0, x.shape[0], dtype=torch.long).cuda()
        pos_embedding = self.position_encode(self.position_index.unsqueeze(0).expand(x.shape[0], x.shape[1]))
        # x = x + pos_embedding

        attn_out, attn_w = self.attn(x, x, x, dist, mask)   # (batch_size, seq_len, emb_dim)
        attn_out = x + self.dropout1(attn_out)
        attn_out = self.layernorm1(attn_out)
        ffn_out = self.linear2(self.dropout(torch.relu(self.linear1(attn_out))))
        out = attn_out + self.dropout2(ffn_out)
        out = self.layernorm2(out)
        return out   # (batch_size, seq_len, emb_dim)


class MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.emb_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.a = nn.Parameter(torch.randn(1))

        # bound = 1.0 / math.sqrt(self.emb_dim)
        # self.W_q = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))
        # self.W_k = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))
        # self.Wt = nn.Parameter(torch.randn(1, self.emb_dim))
        # self.W_q = torch.nn.init.uniform_(self.W_q, -bound, bound)
        # self.W_q = torch.nn.init.uniform_(self.W_q, -bound, bound)
        # self.Wt = torch.nn.init.uniform_(self.Wt, -bound, bound)

        self.wq = nn.Linear(self.emb_dim, self.emb_dim)
        self.wk = nn.Linear(self.emb_dim, self.emb_dim)
        self.wv = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.emb_dim)

    def create_padding_mask(self, seq_mask):
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, seq_len) --> (batch_size, 1, 1, seq_len)
        seq_mask = seq_mask.expand(-1, self.num_heads, seq_mask.size(-1), -1)
        return seq_mask  # mask: (batch_size, seq_len) ---> padding ---> (batch_size, num_heads, seq_len, seq_len)

        # seq_mask1 = seq_mask.unsqueeze(1).unsqueeze(1)   # (batch_size, seq_len) --> (batch_size, 1, 1, seq_len)
        # seq_mask1 = seq_mask1.expand(-1, self.num_heads, seq_mask1.size(-1), -1)  # mask: (batch_size, seq_len) ---> padding ---> (batch_size, num_heads, seq_len, seq_len)
        # seq_mask2 = seq_mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, seq_len, 1)
        # seq_mask2 = seq_mask2.expand(-1, self.num_heads, -1, seq_mask2.shape[2])  # (batch_size, num_heads, seq_len, seq_len)
        # seq_mask = seq_mask1.mul(seq_mask2)
        # 去除对角线上的元素
        # diag = torch.eye(seq_mask.shape[-1], seq_mask.shape[-1]).cuda()
        # diag = diag.expand(seq_mask.shape[0], -1, -1).unsqueeze(1)
        # seq_mask = seq_mask.masked_fill(diag == 1, 0)
        # return seq_mask

    def split_head(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)   # (batch_size, seq_len, num_heads, head_dim) (B, S, N, H)
        return x.transpose(1,2)   # (batch_size, num_heads, seq_len, head_dim) (B, N, S,H)

    def scaled_dot_product_attention(self, q, k, v, dist, mask=None):
        # ==========add attention ============
        # attention = torch.tanh()
        # =========scale dot product==========
        attention = torch.matmul(q, k.transpose(2, 3))   # (batch_size, num_heads, seq_len, seq_len)  (B, N, S, S)
        # Scale
        scale = v.size(-1) ** -0.5
        attention = attention * scale + 0.4 * dist
        # Mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention_weights, v)
        return out, attention_weights    # out: (batch_size, num_heads, seq_len, head_dim)  (B, N, S, H)

    def forward(self, query, key, value, dist, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        # Line Project
        q = self.wq(query)  # (batch_size, seq_len, d_model) (B, S, D)
        k = self.wk(key)    # (batch_size, seq_len, d_model)
        v = self.wv(value)  # (batch_size, seq_len, d_model)
        if mask is not None:
            mask = self.create_padding_mask(mask)  # (batch_size, seq_len) -> padding -> (batch_size, num_heads, seq_len, seq_len)
            # print(mask)

        # Split multil heads
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        dist = dist.unsqueeze(1)
        dist = dist.expand(batch_size, self.num_heads, seq_len, seq_len)

        attention, attention_weights = self.scaled_dot_product_attention(q, k, v, dist, mask)

        attention = attention.transpose(1, 2)    # (batch_size, seq_len, num_heads, head_dim)
        concat_attention = attention.reshape(batch_size, seq_len, self.emb_dim)    # (batch_size, seq_len, emb_dim)
        output = self.linear(concat_attention)

        return output, attention_weights

class Model_Relation_Self_Attention(nn.Module):
    def __init__(self, emb_dim, num_heads, seq_len, max_relative_position, dim_feedforward=64, dropout=0.0):
        super(Model_Relation_Self_Attention, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        # ============positional embedding=====================
        # self.position_encode = nn.Embedding(seq_len, emb_dim)
        # self.position_index = torch.arange(0, seq_len, dtype=torch.long).cuda()
        self.attn = Relation_MultiheadAttention(emb_dim, num_heads, max_relative_position)

        # Position-wise-Feed-Forward-Networks
        self.linear1 = nn.Linear(emb_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, emb_dim)

        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dist, delatimes, mask=None, flag='train'):
        attn_out, attn_w = self.attn(x, x, x, dist, delatimes, mask)   # (batch_size, seq_len, emb_dim)
        attn_out = x + self.dropout1(attn_out)
        attn_out = self.layernorm1(attn_out)
        ffn_out = self.linear2(self.dropout(torch.relu(self.linear1(attn_out))))
        out = attn_out + self.dropout2(ffn_out)
        out = self.layernorm2(out)
        return out   # (batch_size, seq_len, emb_dim)


class Relation_MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, max_relative_position):
        super(Relation_MultiheadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        self.head_dim = emb_dim // num_heads
        assert self.emb_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # ==============relative_positions_embeddings============
        vocab_size = 2 * max_relative_position + 1
        self.embeddings_position_keys = nn.Embedding(vocab_size, self.head_dim)
        # self.embeddings_position_values = nn.Embedding(vocab_size, self.head_dim) 
        self.a = nn.Parameter(torch.randn(1))

        self.wq = nn.Linear(self.emb_dim, self.emb_dim)
        self.wk = nn.Linear(self.emb_dim, self.emb_dim)
        self.wv = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.emb_dim)

    def create_padding_mask(self, seq_mask):
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, seq_len) --> (batch_size, 1, 1, seq_len)
        seq_mask = seq_mask.expand(-1, self.num_heads, seq_mask.size(-1), -1)
        return seq_mask  # mask: (batch_size, seq_len) ---> padding ---> (batch_size, num_heads, seq_len, seq_len)

        # seq_mask1 = seq_mask.unsqueeze(1).unsqueeze(1)   # (batch_size, seq_len) --> (batch_size, 1, 1, seq_len)
        # seq_mask1 = seq_mask1.expand(-1, self.num_heads, seq_mask1.size(-1), -1)  # mask: (batch_size, seq_len) ---> padding ---> (batch_size, num_heads, seq_len, seq_len)
        # seq_mask2 = seq_mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, seq_len, 1)
        # seq_mask2 = seq_mask2.expand(-1, self.num_heads, -1, seq_mask2.shape[2])  # (batch_size, num_heads, seq_len, seq_len)
        # seq_mask = seq_mask1.mul(seq_mask2)
        # 去除对角线上的元素
        # diag = torch.eye(seq_mask.shape[-1], seq_mask.shape[-1]).cuda()
        # diag = diag.expand(seq_mask.shape[0], -1, -1).unsqueeze(1)
        # seq_mask = seq_mask.masked_fill(diag == 1, 0)
        # return seq_mask

    def split_head(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)   # (batch_size, seq_len, num_heads, head_dim) (B, S, N, H)
        return x.transpose(1, 2)   # (batch_size, num_heads, seq_len, head_dim) (B, N, S,H)

    def generate_relative_positions_embeddings(self, seq_len, max_relative_position, name):
        relative_positions_matrix = self.generate_relative_positions_matrix(seq_len, max_relative_position)
        # vocab_size = max_relative_position * 2 + 1
        # embeddings_table = nn.Embedding(vocab_size, emb_dim)
        # embeddings = embeddings_table(relative_positions_matrix)
        if name == 'relative_positions_keys':
            embeddings = self.embeddings_position_keys(relative_positions_matrix)
        if name == 'relative_positions_values':
            embeddings = self.embeddings_position_keys(relative_positions_matrix)
        return embeddings

    def generate_relative_times_embeddings(self, delatimes, seq_len, max_relative_position, name):
        batch = delatimes.shape[0]
        relative_times_matrix = delatimes[:, None] - delatimes.reshape((batch, seq_len, 1))  # batch x seq_len x seq_len
        mat_clipped = torch.clamp(relative_times_matrix, -max_relative_position, max_relative_position)
        final_mat = mat_clipped + max_relative_position
        if name == 'relative_positions_keys':
            embeddings = self.embeddings_position_keys(final_mat)
        if name == 'relative_positions_values':
            embeddings = self.embeddings_position_keys(final_mat)
        return embeddings

    def generate_relative_positions_matrix(self, seq_len, max_relative_position):
        """Generates matrix of relative positions between inputs"""
        range_vec = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
        distance_mat = range_vec[None, :] - range_vec[:, None]  # [seq_len x seq_len]
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat.type(T.LongTensor)

    def relative_attention_inner(self, x, y, z, transpose):
        batch_size = x.shape[0]
        seq_len = z.shape[0]
        # xy_weight ---> (batch_size, num_heads, seq_len, seq_len) (B, N, S, S)
        if transpose:
            xy_weight = torch.matmul(x, y.transpose(2, 3))
        else:
            xy_weight = torch.matmul(x, y)
        # x_v ---> (seq_len, batch_size*num_heads, head_dim) (S, B*N, H)
        x_v = x.reshape(seq_len, batch_size * self.num_heads, -1)
        # xz_weight ---> (seq_len, batch_size*num_heads, S) (S, B*N, S)
        if transpose:
            xz_weight = torch.matmul(x_v, z.transpose(1, 2))
        else:
            xz_weight = torch.matmul(x_v, z)
        # xz_weight ---> (batch_size, num_heads, seq_len, seq_len) (B, N, S, S)
        xz_weight = xz_weight.reshape(batch_size, self.num_heads, seq_len, -1)
        return xy_weight + xz_weight

    def relative_time_attention_inner(self, x, y, z, transpose):
        # x : B, H, S, D
        # z : B, S, S, D
        batch_size = x.shape[0]
        seq_len = z.shape[0]
        # xy_weight ---> (batch_size, num_heads, seq_len, seq_len) (B, N, S, S)
        if transpose:
            xy_weight = torch.matmul(x, y.transpose(2, 3))
        else:
            xy_weight = torch.matmul(x, y)
        # x_v ---> (seq_len, batch_size*num_heads, head_dim) (S, B*N, H)
        # x_v = x.reshape(seq_len, batch_size * self.num_heads, -1)
        # xz_weight ---> (seq_len, batch_size*num_heads, S) (S, B*N, S)
        if transpose:
            xz_weight = torch.matmul(x.transpose(1, 2), z.transpose(2, 3))  # (B,S,H,S)
        else:
            xz_weight = torch.matmul(x.transpose(1, 2), z)
        # xz_weight ---> (batch_size, num_heads, seq_len, seq_len) (B, N, S, S)
        # xz_weight = xz_weight.reshape(batch_size, self.num_heads, seq_len, -1)
        xz_weight = xz_weight.transpose(1, 2)
        return xy_weight + xz_weight

    def forward(self, query, key, value, dist, delatimes, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
		
        relations_keys = self.generate_relative_times_embeddings(delatimes, seq_len, self.max_relative_position,
                                                                          name='relative_positions_keys')  # (S, S, D)
        relations_values = relations_keys
        # Line Project
        q = self.wq(query)  # (batch_size, seq_len, d_model) (B, S, D)
        k = self.wk(key)    # (batch_size, seq_len, d_model)
        v = self.wv(value)  # (batch_size, seq_len, d_model)
        if mask is not None:
            mask = self.create_padding_mask(mask)  # (batch_size, seq_len) -> padding -> (batch_size, num_heads, seq_len, seq_len)

        # Split multil heads
        q = self.split_head(q)  # (batch_size, num_heads, seq_len, head_dim) (B, N, S, H)
        k = self.split_head(k)  # (batch_size, num_heads, seq_len, head_dim) (B, N, S, H)
        v = self.split_head(v)  # (batch_size, num_heads, seq_len, head_dim) (B, N, S, H)

        dist = dist.unsqueeze(1)
        dist = dist.expand(batch_size, self.num_heads, seq_len, seq_len)
        scale = q.size(-1) ** -0.5
        weight = self.relative_time_attention_inner(q, k, relations_keys, transpose=True)

        weight = weight * scale + 0.6 * dist 
        if mask is not None:
            weight = weight.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(weight, dim=-1)
        # out (batch_size, head_nums, seq_len, mb_dim) (B, N, S, H)
        attention = self.relative_time_attention_inner(attention_weights, v, relations_values, transpose=False)
        # attention = torch.matmul(attention_weights, v)
        attention = attention.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        concat_attention = attention.reshape(batch_size, seq_len, self.emb_dim)  # (batch_size, seq_len, emb_dim)
        output = self.linear(concat_attention)
        return output, attention_weights


class STSA(nn.Module):
    def __init__(self, config):
        super(STSA, self).__init__()
        self.config = config
        self.user_nums = config['user_nums']
        self.item_nums = config['item_nums']
        self.emb_dim = config['emb_dim']
        self.attention_dim = config['attention_dim']
        self.key_dim = config['key_dim']
        self.value_dim = config['value_dim']
        self.slot_num = config['slot_num']
        self.hop_num = config['hop_num']
        self.alpha = config['alpha']
        self.dropout_rate = config['dropout']
        self.model = config['model']
        self.short_method = config['short_method']
        self.long_method = config['long_method']
        self.co_attention = config['co_attention']
        self.scale = self.emb_dim ** -0.5

        self.embedding_user = nn.Embedding(self.user_nums, self.emb_dim)
        self.embedding_item = nn.Embedding(self.item_nums, self.emb_dim)
        self.embedding_time = nn.Embedding(48, self.emb_dim)

        bound = 1.0 / math.sqrt(self.emb_dim)

        self.Key = nn.Parameter(torch.randn(self.slot_num, self.key_dim))
        # self.Value = nn.Parameter(torch.randn(self.slot_num, self.value_dim))
        self.Value = nn.Parameter(torch.randn(self.user_nums, self.slot_num, self.value_dim))
        # self.Key = torch.nn.init.uniform_(self.Key, -bound, bound)
        # self.Value = torch.nn.init.uniform_(self.Value, -bound, bound)
        # self.embedding_user.weight.data.normal_(0, 1.0 / self.emb_dim)
        # self.embedding_item.weight.data.normal_(0, 1.0 / self.emb_dim)
        self.Key = torch.nn.init.normal_(self.Key, 0, 1.0)
        self.Value = torch.nn.init.normal_(self.Value, 0, 1.0)

        if config['short_method'] == 'Self_Attention':
            # self.Short_Model = Model_Self_Attention(config['emb_dim'], config['num_heads'],config['seq_len'],
            #                                         config['dim_feedforward'], config['dropout'])
            self.Short_Model = Model_Relation_Self_Attention(config['emb_dim'], config['num_heads'], config['seq_len'],
                                                             config['max_relative_position'],
                                                             config['dim_feedforward'], config['dropout'])
        else:
            self.Short_Model = Model_RNN(config)

        self.Memory_item = MANN_Item(config)
        self.Memory_feature = MANN_Feature(config)

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

        # Paralla Co-attention
        self.W_c = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim))

    def forward(self, users, seq_items, seq_times, hist_items, seq_dist, hist_dist, target_items, target_times, delatimes, s_mask=None, h_mask=None, flag='train'):
        if self.model == 'model_long':
            user = self.embedding_user(users)
            target = self.embedding_item(target_items)
            target_time = self.embedding_time(target_times)
            if self.long_method == 'item':
                hist = self.embedding_item(hist_items)
                p_l = self.Memory_item(target, hist, self.scale, h_mask, flag=flag)
                # p_l = p_l + user
            else:
                p_l = self.Memory_feature(users, target, flag=flag)
            pred_y = torch.sum(p_l.mul(target), dim=1) + torch.sum(target_time.mul(target), dim=1)
            pred_y = torch.sigmoid(pred_y)
            return pred_y


        elif self.model == 'model_short':
            #user = self.embedding_user(users)    # (B, D)
            seq = self.embedding_item(seq_items)  # (B, S, D)
            #seq_time = self.embedding_time(seq_times)
            # seq_dist[seq_dist > 0] = 1
            seq_dist = T.FloatTensor(seq_dist)
            target = self.embedding_item(target_items)
            target_time = self.embedding_time(target_times)
            if flag == 'train':
                out_short = self.Short_Model(seq, seq_dist, delatimes, s_mask, flag=flag)  # (batch_size, seq_len, emb_dim)  # Transformer
                # out_short = self.Short_Model(seq, seq_dist, s_mask, flag=flag)
                # mask = s_mask.float().unsqueeze(-1)
                # out_short = mask.mul(out_short)
                # out_short, hn = self.Short_Model(seq, s_mask, flag=flag)  # (batch_size, seq_len, emb_dim)   # RNN
            else:
                out_short = self.Short_Model(seq.unsqueeze(0), seq_dist.unsqueeze(0), delatimes.unsqueeze(0), s_mask, flag=flag)
                # out_short = self.Short_Model(seq.unsqueeze(0), seq_dist.unsqueeze(0), flag=flag)
                # out_short, hn = self.Short_Model(seq.unsqueeze(0), s_mask, flag=flag)  # (batch_size, seq_len, emb_dim)   # RNN

            #-----------------------------Loction attention------------------------
            att_l = target.unsqueeze(2)
            weight_l = torch.matmul(out_short, att_l) #* self.scale
            if flag == 'train':
                s_mask = s_mask.unsqueeze(2)
                weight_l = weight_l.masked_fill(s_mask == 0, -1e9)
            weight_l = torch.softmax(weight_l, dim=1)
            out_l = torch.sum(weight_l.mul(out_short), dim=1)  # n x d

            # -----------------------------Time attention------------------------
            # att_t = target_time.unsqueeze(2)
            # weight_t = torch.matmul(out_short, att_t) * self.scale
            # weight_t = torch.softmax(weight_t, dim=1)
            # out_t = torch.sum(weight_t.mul(out_short), dim=1)  # n x d

            # weight = weight_l + weight_t
            # weight = torch.softmax(weight, dim=1)
            # out = torch.sum(weight.mul(out_short), dim=1)  # n x d

            # ----------------------concat-----------------------
            # out = torch.cat((out_l, out_t), dim=-1)
            # out = self.co_attn(out)
            # out = self.dropout(out)

            # ---------------------weighted----------------------
            # out = torch.add(torch.matmul(out_l, self.w_attn_l) + torch.matmul(out_t, self.w_attn_t), self.bias)

            # ---------------------add---------------------------
            # out = out_t + out_l
            #out = out_l

            pred_y = torch.sum(out.mul(target), dim=1) + torch.sum(target_time.mul(target), dim=1)
            pred_y = torch.sigmoid(pred_y)
            # pred_y = torch.sigmoid(torch.sum(out.mul(target), dim=1))   # user attention + location attention

            return pred_y

        elif self.model == 'model_fuse':
            # user = self.embedding_user(users)
            seq = self.embedding_item(seq_items)
            seq_dist = T.FloatTensor(seq_dist)
            # seq_time = self.embedding_time(seq_times)
            # hist = self.embedding_item(hist_items)
            target = self.embedding_item(target_items)
            target_time = self.embedding_time(target_times)

            if self.co_attention:
                if flag == 'train':
                    out_h = self.Short_Model(seq, seq_dist, delatimes, s_mask, flag=flag)  # (batch_size, seq_len, emb_dim)
                    # out_h = self.Short_Model(seq, seq_dist, s_mask, flag=flag)
                    s_mask = s_mask.unsqueeze(2)  # (batch_size, seq_len, 1)
                    # ======================RNN====================
                    # out_h, hn = self.Short_Model(seq, s_mask, flag=flag)  # (batch_size, seq_len, emb_dim)   # RNN
                    # s_mask = s_mask.unsqueeze(2)  # (batch_size, seq_len, 1)

                    # att_l = target.unsqueeze(2)
                    # weight_l = torch.matmul(out_h, att_l) * self.scale
                    # weight_l = weight_l.masked_fill(s_mask == 0, -1e9)
                    # weight_l = torch.softmax(weight_l, dim=1)
                    # h0 = torch.sum(weight_l.mul(out_h), dim=1)  # n x d

                    out_h = s_mask.float().mul(out_h)
                    # h0 = torch.sum(out_h, dim=1) / s_mask.sum(dim=1).float()  # out_h.shape[1]  # (batch_size, emb_dim)
                else:
                    out_h = self.Short_Model(seq.unsqueeze(0), seq_dist.unsqueeze(0), delatimes.unsqueeze(0), s_mask, flag=flag)
                    # out_h = self.Short_Model(seq.unsqueeze(0), seq_dist.unsqueeze(0), flag=flag)
                    # out_h = out_h.expand(target.shape[0], -1, -1)
                    # =======================RNN=======================
                    # out_h, hn = self.Short_Model(seq.unsqueeze(0), s_mask, flag=flag)  # (batch_size, seq_len, emb_dim)   # RNN

                    # att_l = target.unsqueeze(2)
                    # weight_l = torch.matmul(out_h, att_l) * self.scale
                    # weight_l = torch.softmax(weight_l, dim=1)
                    # h0 = torch.sum(weight_l.mul(out_h), dim=1)  # n x d

                    # h0 = torch.sum(out_h, dim=1) / out_h.shape[1]  # (batch_size, emb_dim)

                # ======================Parall Co-attention1======================
                # S = out_h.expand(target.shape[0], -1, -1)  # (B,  S, D)
                if flag == 'test':
                    S = out_h.expand(target.shape[0], -1, -1).transpose(1, 2)   # (B,  D, S)
                else:
                    S = out_h.transpose(1, 2)   # (B,  D, S)
                # K = self.Key.unsqueeze(0).expand(S.shape[0], -1, -1).transpose(1, 2)  # (B,  K, D)
                # M = self.Value.unsqueeze(0).expand(S.shape[0], -1, -1)  # .transpose(1, 2)  # (B,  K, D)
                K = self.Value[users].transpose(1, 2)  # (B,  D, K)========================================

                # C = torch.tanh(torch.matmul(S.transpose(1, 2), torch.matmul(self.W_c, K)))  # (B, S, K)
                C = torch.matmul(S.transpose(1, 2), torch.matmul(self.W_c, K))  # (B, S, K) ============================
                # C = torch.matmul(S.transpose(1, 2), K)  # (B, S, K) 不带参数
                #print(C)
				
				# ============= dot ===================
                a_s = torch.sum(torch.mul(S, target.unsqueeze(2).expand(-1, -1, S.shape[2])), dim=1).unsqueeze(1) + \
                      torch.sum(torch.mul(S, torch.matmul(K, torch.softmax(C.transpose(1, 2), dim=2))), dim=1).unsqueeze(1)  # (B, 1, S)===========================
                a_m = torch.sum(torch.mul(K, target.unsqueeze(2).expand(-1, -1, K.shape[2])), dim=1).unsqueeze(1) + \
                      torch.sum(torch.mul(K, torch.matmul(S, torch.softmax(C, dim=2))), dim=1).unsqueeze(1)  # (B, 1, K)

                # ========cat, convert=========
                # q_s = torch.cat((target.unsqueeze(2).expand(-1, -1, S.shape[2]), torch.matmul(K, C.transpose(1, 2))), dim=1)  # (B,  2D, S)
                # # q_s = torch.cat((target.unsqueeze(1).expand(-1, -1, S.shape[1]), torch.matmul(torch.softmax(C,dim=2), K)), dim=-1)
                # q_s = torch.matmul(self.tranH.t(), q_s)
                # q_s = target.unsqueeze(2).expand(-1, -1, S.shape[2])
                #
                # q_m = torch.cat((target.unsqueeze(2).expand(-1, -1, K.shape[2]), torch.matmul(S, C)), dim=1)   # (B, 2D, K)
                # # q_m = torch.cat((target.unsqueeze(1).expand(-1, K.shape[1], -1), torch.matmul(torch.softmax(C.transpose(1, 2), dim=2), S)), dim=-1)
                # q_m = torch.matmul(self.tranM.t(), q_m)  # (B, K, D)s
                # # q_m = torch.matmul(C.transpose(1, 2), S)   # (B, K, D)
                # q_m = target.unsqueeze(2).expand(-1, -1, K.shape[2])

                # ========weight, convert========
                # q_s = torch.matmul(torch.matmul(self.W_l, K), C.transpose(1, 2)) + \
                #       torch.matmul(self.W_t, target.unsqueeze(-1)).expand(-1, -1, S.shape[-1]) # (B,  D, S)
                # q_m = torch.matmul(torch.matmul(self.W_s, S), C) + \
                #       torch.matmul(self.W_t, target.unsqueeze(-1)).expand(-1, -1, K .shape[-1])  # (B,  D, K)
                #
                # # q_s = torch.matmul(torch.matmul(self.W_l, K), torch.softmax(C.transpose(1, 2), dim=1)) + \
                # #       torch.matmul(self.W_t, target.unsqueeze(-1)).expand(-1, -1, S.shape[-1])  # (B,  D, S)
                # # q_m = torch.matmul(torch.matmul(self.W_s, S), torch.softmax(C, dim=1)) + \
                # #       torch.matmul(self.W_t, target.unsqueeze(-1)).expand(-1, -1, K.shape[-1])  # (B,  D, K)
                #
                # a_s = torch.sum(torch.mul(S, q_s), dim=1).unsqueeze(1)  # (B, 1, S)
                # a_m = torch.sum(torch.mul(K, q_m), dim=1).unsqueeze(1)  # (B, 1, K)
                
                if flag == 'train':
                    a_s = a_s.masked_fill(s_mask.transpose(1, 2) == 0, -1e9)
                a_s = torch.softmax(a_s, dim=2)  # (B, 1, S)
                a_m = torch.softmax(a_m, dim=2)  # (B, 1, K)

                out_short = torch.squeeze(torch.matmul(a_s, S.transpose(1, 2)))
                out_long = torch.squeeze(torch.matmul(a_m, K.transpose(1, 2)))
              
                out = self.alpha * out_long + (1 - self.alpha) * out_short

                pred_y = torch.sum(out.mul(target), dim=1) + torch.sum(target_time.mul(target), dim=1)# + cs
                pred_y = torch.sigmoid(pred_y)
                return pred_y


class Train_Model(object):
    def __init__(self, config):
        self.config = config
        self.model = STSA(config)
        if torch.cuda.is_available():
            self.model.cuda()
        if config['loss_fuction'] == 'BPRLoss':
            self.criterion = BPRLoss()
        else:
            self.criterion = torch.nn.BCELoss()

        self.optimizer = self.get_optimizer()

    def get_optimizer(self):
        if self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config['sgd_lr'],
                                        momentum=self.config['sgd_momentum'],
                                        weight_decay=self.config['l2_regularization'])
        elif self.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.config['adam_lr'],
                                         weight_decay=self.config['l2_regularization'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.config['rmsprop_lr'],
                                            alpha=self.config['rmsprop_alpha'],
                                            momentum=self.config['rmsprop_momentum'],
                                            weight_decay=self.config['l2_regularization'])
        return optimizer

    def train_batch(self, users, seq_items, seq_times, hist_items, seq_dist, hist_dist, target_items, target_times, delatimes, labels, s_mask, h_mask):
        self.optimizer.zero_grad()
        pred_y= self.model(users, seq_items, seq_times, hist_items, seq_dist, hist_dist, target_items, target_times, delatimes, s_mask, h_mask, flag='train')
        loss = self.criterion(pred_y.view(-1), labels)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return loss

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), model_dir)


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = -torch.log(torch.sigmoid(x - y))
        return loss

