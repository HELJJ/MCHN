import torch
from torch import nn
import pdb
import numpy as np
from scipy.sparse import coo_matrix
from models.layers import EmbeddingLayer, GraphLayer, TransitionLayer, Med_EmbeddingLayer
from models.utils import DotProductAttention, SoftAttention, visit_DotProductAttention, hypergraph_part, pairgraph_part, VisitRepresentation
# from torch_geometric.nn import HypergraphConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from torch_geometric.nn import HypergraphConv
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0., activation=None):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


def dia_med_hypergraph(c_it, med_it):
    disease_index = torch.nonzero(c_it).squeeze().tolist()
    medicine_index = torch.nonzero(med_it).squeeze().tolist()
    dia_len = len(disease_index)
    medicine_node_index = list(i+dia_len for i in range(0, len(medicine_index)))

    diagnosis_node_index = list(range(0, len(disease_index)))
    dia_med_node_index = []
    dia_med_edge_index = []
    count = 0
    for dia_h in diagnosis_node_index:
        dia_med_node_index = dia_med_node_index + dia_h + medicine_node_index
        dia_med_edge_index = dia_med_edge_index + len(dia_med_node_index)*[count]
        count = count + 1

    hypergraph_data = np.array(list(len(dia_med_node_index)*[1]))
    #行表示超边，列表示节点
    dia_med_edge_num = len(dia_med_edge_index) #超边数量
    dia_med_node_num = len(dia_med_node_index) #节点数量
    dia_med_edge_index = np.array(dia_med_edge_index)
    dia_med_node_index = np.array(dia_med_node_index)
    hypergraph = coo_matrix((hypergraph_data, (dia_med_edge_index, dia_med_node_index)), shape=(dia_med_edge_num, dia_med_node_num))
    print('dia_med_node_index len is %d, dia_med_edge_index len is %d'%(len(dia_med_node_index), len(dia_med_edge_index)))
    # 转换为torch.sparse.Tensor
    values = hypergraph.data
    indices = np.vstack((hypergraph.row, hypergraph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = hypergraph.shape
    hypergraph = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return hypergraph


def dia_med_hypergraphs(c_it, med_it, c_embeddings, m_embeddings):
    # Get indices of non-zero elements (i.e., diagnosed diseases and prescribed medications)
    disease_indices = torch.nonzero(c_it).squeeze().tolist()
    med_indices = torch.nonzero(med_it).squeeze().tolist()
    # Extract embeddings for diseases and medications
    disease_embeddings = c_embeddings[disease_indices]
    med_embeddings = m_embeddings[med_indices]
    # Combine disease and medication embeddings
    combined_embeddings = torch.cat((disease_embeddings, med_embeddings), dim=0)
    # Create hyperedges
    hyperedges = []
    for disease in disease_indices:
        hyperedges.append([disease] + med_indices)
        # Initialize hyperedge attributes as learnable parameters
        hyperedge_attr = nn.Parameter(torch.randn(len(hyperedges), combined_embeddings.shape[1]))
    return hyperedges, combined_embeddings, hyperedge_attr

def concat_representations(dia_med_node_feature, dia_node_feature, med_node_feature, c_it, med_it):
    # Get the number of diseases and medicines from the input tensors
    num_diseases = torch.nonzero(c_it).size(0)
    num_meds = torch.nonzero(med_it).size(0)
    # Assuming each disease in dia_med_node_feature has a representation combined with each medicine
    # We'll average these combined representations to get a single representation for each disease and medicine
    dia_med_disease_rep = dia_med_node_feature[:num_diseases*num_meds].view(num_diseases, num_meds, -1).mean(dim=1)
    dia_med_medicine_rep = dia_med_node_feature[num_diseases*num_meds:].view(num_meds, num_diseases, -1).mean(dim=1)
    # Concatenate the representations
    disease_final_rep = torch.cat([dia_med_disease_rep, dia_node_feature], dim=1)
    medicine_final_rep = torch.cat([dia_med_medicine_rep, med_node_feature], dim=1)
    final_representation = torch.cat([disease_final_rep, medicine_final_rep], dim=0)
    return disease_final_rep, medicine_final_rep, final_representation

def compute_tf_idf(patients_diagnosis_matrix):

    return tfidf_matrix


def construct_graph_topology(tfidf_matrix, k):
    cosine_sim = cosine_similarity(tfidf_matrix)
    knn = NearestNeighbors(n_neighbors=k).fit(cosine_sim)
    adjacency_matrix = knn.kneighbors_graph(cosine_sim).toarray()
    return adjacency_matrix



class Model(nn.Module): #重要参数：code_size; hidden_size 256; output_size; trans_embedding_dim
    def __init__(self, code_num, code_size,
                 adj, graph_size, hidden_size, trans_embedding_dim, t_attention_size, t_output_size, #graph_size; t_attention_size; t_output_size用不到
                 output_size, dropout_rate, activation, med_code_num):
        super().__init__()
        in_channels, out_channels = (code_size, hidden_size)
        # self.in_channels = in_channels
        self.embedding_layer = EmbeddingLayer(code_num, code_size, graph_size)
        self.med_embedding_layer = Med_EmbeddingLayer(med_code_num, code_size, graph_size)
        self.graph_layer = GraphLayer(adj, code_size, graph_size)
        self.transition_layer = TransitionLayer(code_num, graph_size, hidden_size, t_attention_size, t_output_size)
        self.attention = DotProductAttention(3*hidden_size, 32)#32
        self.classifier = Classifier(3*hidden_size, output_size, dropout_rate, activation) #output_size
        self.conv = HypergraphConv(in_channels, out_channels)
        self.conv_gat = HypergraphConv(in_channels, out_channels, use_attention = True)
        self.soft_attetion = SoftAttention(out_channels, hidden_size)
        self.visit_attention = visit_DotProductAttention(hidden_size, hidden_size)
        self.hyperedge_attr = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(code_num, in_channels)))
        self.hypergraph_encode = hypergraph_part(in_channels, out_channels, hidden_size)
        self.pair_graph_encode = pairgraph_part(code_size,trans_embedding_dim)
        self.visit_representation_module = VisitRepresentation(hidden_size=hidden_size, attention_dim=hidden_size, output_dim=3*hidden_size)
    def forward(self, code_x, divided, neighbors, lens, medicine_codes):
            #f1_score: 0.2048 --- top_k_recall: 0.2027, 0.2902, chet
            #0.2268
            #0.1314
            #0.2623 甚至会更高 medication与disease合并起来，然后不进行linear
            #0.19148 medication与disease以及med-dis的超图合并起来，linear bias=True
            #0.170 medication与disease以及med-dis的超图合并起来，linear bias=False
            #f1_score: 0.2691 --- top_k_recall: 0.2523, 0.3593,  (512,)
            #hypergraph+pairgraph (256,) f1_score: 0.1242 --- top_k_recall: 0.1489, 0.2636
            #hypergraph+pairgraph hypergraph是512,pairgraph是256, 合并起来之后不加dense，对所有visit进行attention之后得到的事768的向量，f1_score: 0.2692 --- top_k_recall: 0.2549, 0.3649
            #f1_score: 0.2598 --- top_k_recall: 0.2483, 0.3572, 只有diagnosis channel，消融med通道以及dia-med通道
            #1_score: 0.1339 --- top_k_recall: 0.1877, 0.2750, 保留dia以及dia-med通道，消融掉med通道
            #                                                   消融共现通道
        use_cuda = True
        device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        embeddings = self.embedding_layer()
        med_embeddings = self.med_embedding_layer()
        c_embeddings, n_embeddings, u_embeddings = embeddings
        m_embeddings = med_embeddings
        output = [] #不同的患者
        for code_x_i, divided_i, neighbor_i, len_i, medicine_i in zip(code_x, divided, neighbors, lens, medicine_codes):#遍历患者
            '''
            tfidf_matrix = compute_tf_idf(patients_diagnosis_matrix)
            
            construct_graph_topology(tfidf_matrix, k=5) #K是个超参，后面后主函数传入
            '''
            no_embeddings_i_prev = None
            output_i = []
            h_t = None
            for t, (c_it, d_it, n_it, len_it, medicine_it) in enumerate(zip(code_x_i, divided_i, neighbor_i, range(len_i), medicine_i)): #遍历当前就诊诊断出的疾病
                co_embeddings, no_embeddings = self.graph_layer(c_it, n_it, c_embeddings, n_embeddings)
                transgraph_repre, h_t = self.transition_layer(t, co_embeddings, d_it, no_embeddings_i_prev, u_embeddings, h_t)
                no_embeddings_i_prev = no_embeddings
                hypergraph_repre = self.hypergraph_encode(c_it, medicine_it, c_embeddings, m_embeddings)
                pair_repre = self.pair_graph_encode(c_it, medicine_it, c_embeddings, m_embeddings)
                #hypergraph_repre pair_repre transgraph_repre 这三个向量维度都一样
                visit_representation = self.visit_representation_module(hypergraph_repre, pair_repre, transgraph_repre) # visit_representation代表一次就诊的表示
                #output_i.append(visit_representation) #
                output_i.append(hypergraph_repre)  #得到的哪个通道的表示
            output_i = self.attention(torch.vstack(output_i)) #患者历史就诊表示做注意力
            output.append(output_i)
        output = torch.vstack(output)
        output = self.classifier(output)
        return output
