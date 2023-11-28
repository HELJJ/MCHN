import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, value, key, query, mask=None):
        attention_out, attention_weights, _ = self.attention(query, key, value, attn_mask=mask)
        x = self.norm1(attention_out + query)
        forward_out = self.feed_forward(x)
        out = self.norm2(forward_out + x)
        return out, attention_weights


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, ouput_emb):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, ouput_emb)

    def forward(self, values, keys, query, mask):
        N, _ = values.size()

        # Reshape
        values = values.view(N, self.heads, -1)
        keys = keys.view(N, self.heads, -1)
        queries = query.view(N, self.heads, -1)

        # Ensure the size of the last dimension is correct
        _, _,  value_len = values.size()
        _, _, key_len = keys.size()
        _, _, query_len = queries.size()

        # Split the last dimension into (num_nodes, head_dim)
        values = values.view(N, -1, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, -1, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, -1, self.heads, self.head_dim).transpose(1, 2)

        # Self attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).transpose(1, 2).contiguous().view(N, -1, self.heads * self.head_dim)

        out = self.fc_out(out)
        out = out.squeeze(1)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, trans_embedding_dim):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, trans_embedding_dim)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, trans_embedding_dim),#这个trans_embedding_dim是在第一个linear层变换呢还是在第二个linear层变换呢！！！
            nn.ReLU(),
            nn.Linear(trans_embedding_dim, trans_embedding_dim) #256
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally feed forward network
        out = self.norm1(attention + query)
        out = self.feed_forward(out)
        out = self.norm2(out + attention)
        return out

class GraphEncoder(nn.Module):
    def __init__(self, embed_size, heads, depth, trans_embedding_dim):
        super(GraphEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, trans_embedding_dim) for _ in range(depth)
        ])

    def forward(self, node_embeddings, mask=None):
        out = node_embeddings
        for layer in self.encoder_layers:
            out = layer(out, out, out, mask)
        return out

class pairgraph_part(nn.Module):
    def __init__(self, code_size, trans_embedding_dim):
        super().__init__()
        self.embedding_dim = trans_embedding_dim #超参
        self.graph_encoder = GraphEncoder(embed_size=code_size, heads=8, depth=3, trans_embedding_dim= trans_embedding_dim)
        #self.transformer_layer = TransformerLayer(embed_size=self.embedding_dim, heads=8)  # Adjust as needed
    def forward(self, c_it, medicine_it, c_embeddings, m_embeddings):
        # Get indices of non-zero elements (i.e., diagnosed diseases and prescribed medications)
        disease_indices = torch.nonzero(c_it).squeeze().tolist()
        med_indices = torch.nonzero(medicine_it).squeeze().tolist()
        # Extract embeddings for diseases and medications
        disease_embeddings = c_embeddings[disease_indices]
        med_embeddings = m_embeddings[med_indices]
        # Combine disease and medication embeddings
        combined_embeddings = torch.cat((disease_embeddings, med_embeddings), dim=0)
        # Construct pairwise graph
        nodes = torch.cat([c_it, medicine_it])  # Combining diseases and medicines as nodes
        node_embeddings = combined_embeddings  # Assuming self.embedding is the embedding layer for nodes
        # Use transformer to encode the graph
        #encoded_nodes, attention_weights = self.transformer_layer(node_embeddings, node_embeddings, node_embeddings)
        encoded_nodes = self.graph_encoder(node_embeddings)
        return encoded_nodes

def dia_med_hypergraphs(c_it, med_it, c_embeddings, m_embeddings):
    # Get indices of non-zero elements (i.e., diagnosed diseases and prescribed medications)
    disease_indices = torch.nonzero(c_it).squeeze().tolist()
    med_indices = torch.nonzero(med_it).squeeze().tolist()
    disease_index = list(range(0, len(disease_indices)))
    med_index = [x+len(disease_index) for x in range(0, len(med_indices))]

    # Extract embeddings for diseases and medications
    disease_embeddings = c_embeddings[disease_indices]
    med_embeddings = m_embeddings[med_indices]
    # Combine disease and medication embeddings
    combined_embeddings = torch.cat((disease_embeddings, med_embeddings), dim=0)
    # Create hyperedges
    hyperedges = []
    indicate_edges = []
    count = 0
    hyper_edge_index = []
    for disease in disease_index:
        hyperedges = hyperedges + [disease] + med_index
        indicate_edges = indicate_edges + (len(med_index)+1)*[count]
        count = count+1
        # Initialize hyperedge attributes as learnable parameters
    hyper_edge_index.append(hyperedges)
    hyper_edge_index.append(indicate_edges)
    edge_num = max(hyper_edge_index[1]) + 1
    hyperedge_attr = nn.Parameter(torch.randn(edge_num, combined_embeddings.shape[1]))
    return hyper_edge_index, combined_embeddings, hyperedge_attr

def concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature, c_it, med_it):
    # Get the number of diseases and medicines from the input tensors
    num_diseases = torch.nonzero(c_it).size(0)
    num_meds = torch.nonzero(med_it).size(0)
    # Assuming each disease in dia_med_node_feature has a representation combined with each medicine
    # We'll average these combined representations to get a single representation for each disease and medicine
    dia_med_disease_rep = dia_med_node_feature[:num_diseases]
    dia_med_medicine_rep = dia_med_node_feature[num_diseases:]
    # Concatenate the representations
    disease_final_rep = torch.cat([dia_med_disease_rep, dia_node_feature], dim=1)
    medicine_final_rep = torch.cat([dia_med_medicine_rep, med_node_feature], dim=1)
    final_representation = torch.cat([disease_final_rep, medicine_final_rep], dim=0)
    #final_representation = torch.cat([dia_node_feature, med_node_feature], dim=0)
    return disease_final_rep, medicine_final_rep, final_representation


class hypergraph_part(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels)
        self.conv_gat = HypergraphConv(in_channels, out_channels, use_attention=True , attention_mode='node')
        self.soft_attetion = SoftAttention(out_channels, hidden_size)
        self.linear_layer = nn.Linear(in_features=512, out_features=256, bias=False)

    def forward(self, c_it, medicine_it, c_embeddings, m_embeddings):
        # diagnosis channel 将一次就诊中所有的diagnoses形成一个超图
        diagnosis_hyperedge_index = []
        disease_index = torch.nonzero(c_it).squeeze().tolist()
        if type(disease_index) is int:
            print('This disease_index is int')
            disease_index_list = []
            disease_index_list.append(disease_index)
            disease_index = disease_index_list
            # print('disease_index:',disease_index)
        diagnosis_hyperedge_index.append(list(range(0, len(disease_index))))
        edge = len(disease_index) * [0]
        diagnosis_hyperedge_index.append(edge)
        diagnosis_hyperedge_index = torch.tensor(diagnosis_hyperedge_index).to(c_embeddings.device)
        dia_embedding = c_embeddings[disease_index]
        dia_node_feature = self.conv(dia_embedding, diagnosis_hyperedge_index)
        # medicine channel 将一次就诊中所有的medicine形成一个超图
        medicine_hyperedge_index = []
        medicine_index = torch.nonzero(medicine_it).squeeze().tolist()
        if type(medicine_index) is int:
            print('This disease_index is int')
            medicine_index_list = []
            medicine_index_list.append(medicine_index)
            medicine_index = medicine_index_list
        medicine_hyperedge_index.append(list(range(0, len(medicine_index))))
        edge = len(medicine_index) * [0]
        medicine_hyperedge_index.append(edge)
        medicine_hyperedge_index = torch.tensor(medicine_hyperedge_index).to(c_embeddings.device)
        med_embedding = m_embeddings[medicine_index]  # 需要定义m_embeddings
        med_node_feature = self.conv(med_embedding, medicine_hyperedge_index)

        # diagnosis-medicine channel 将一次就诊中的diangoses与medicine共现关系形成一个超图
        dia_med_hyperedges, dia_med_emb, dia_med_hyperedge_attr = dia_med_hypergraphs(c_it, medicine_it, c_embeddings,
                                                                               m_embeddings)
        dia_med_hyperedges = torch.tensor(dia_med_hyperedges).to(c_embeddings.device)
        dia_med_emb = dia_med_emb.to(c_embeddings.device)
        dia_med_hyperedge_attr = dia_med_hyperedge_attr.to(c_embeddings.device)
        dia_med_node_feature = self.conv_gat(dia_med_emb, dia_med_hyperedges, hyperedge_attr=dia_med_hyperedge_attr) #, hyperedge_attr=dia_med_hyperedge_attr
        #dia_med_node_feature = self.conv(dia_med_emb, dia_med_hyperedges)
        disease_final_rep, medicine_final_rep, final_representation = concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature, c_it, medicine_it)
        #final_representation = self.linear_layer(final_representation)
        return dia_node_feature #返回哪个通道的表示
        #return final_representation


class MedicalCodeAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        super(MedicalCodeAttention, self).__init__()
        self.dense = nn.Linear(hidden_size, attention_dim)
        self.context = nn.Parameter(nn.init.xavier_uniform_(torch.empty(attention_dim, 1)))

    def forward(self, x):
        # x: [node_num, hidden_size]
        attention_weights = torch.tanh(self.dense(x))  # [node_num, attention_dim]
        attention_weights = torch.matmul(attention_weights, self.context).squeeze()  # [node_num]
        attention_weights = F.softmax(attention_weights, dim=-1)
        output = torch.sum(x * attention_weights.unsqueeze(-1), dim=0)  # [hidden_size]
        return output

class VisitRepresentation(nn.Module):
    def __init__(self, hidden_size, attention_dim, output_dim):
        super(VisitRepresentation, self).__init__()
        self.hypergraph_attention = MedicalCodeAttention(2*hidden_size, 2*hidden_size)#attention_dim
        self.pairgraph_attention = MedicalCodeAttention(hidden_size, attention_dim)
        self.dense = nn.Linear(3 * hidden_size, output_dim) #256*4=1024

    def forward(self, hypergraph_repre, pair_repre, transgraph_repre):
        hypergraph_weighted = self.hypergraph_attention(hypergraph_repre) #这是计算hypergraph_repre中节点之间的重要性
        pairgraph_weighted = self.pairgraph_attention(pair_repre) #这是计算pair_repre中节点之间的重要性
        #combined = torch.cat([hypergraph_weighted, pairgraph_weighted, transgraph_repre], dim=-1) #这是将hypergraph_repre得到的就诊表示与pair_repre得到的就诊表示cat起来
        combined = torch.cat([hypergraph_weighted, pairgraph_weighted], dim=-1) #这是将hypergraph_repre得到的就诊表示与pair_repre得到的就诊表示cat起来
        #output = self.dense(combined) #这边将最终的就诊表示接入一个dense层，这一步总感觉,本来combined维度很大，然后突然变成了一个256维度，信息丢失太多吧;试一下不加这个dense与加这个dense的效果
        return combined



class DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output


class visit_DotProductAttention(nn.Module):
    def __init__(self, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_size, 1)))
        self.dense = nn.Linear(value_size, attention_size)

    def forward(self, x):
        t = self.dense(x)
        vu = torch.matmul(t, self.context).squeeze()
        score = torch.softmax(vu, dim=-1)
        output = torch.sum(x * torch.unsqueeze(score, dim=-1), dim=-2)
        return output

class SoftAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SoftAttention, self).__init__()
        self.attention_size = attention_dim
        self.context = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(attention_dim, 1)))
        self.dense = nn.Linear(input_dim, attention_dim)

    def forward(self, inputs):
        # inputs should be of size [batch_size, seq_len, input_dim]
        t = self.dense(inputs)  # [batch_size, seq_len, attention_dim]
        #attention_weights = torch.tanh(attention_weights)  # [batch_size, seq_len, attention_dim]
        attention_weights = torch.matmul(t, self.context).squeeze()  # [batch_size, seq_len]
        score = torch.softmax(attention_weights, dim=-1)
        output = torch.sum(inputs * torch.unsqueeze(score, dim=-1), dim=-2)
        return output
