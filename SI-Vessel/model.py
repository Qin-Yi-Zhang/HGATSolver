import torch
import torch.nn as nn
import math
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.data import HeteroData

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_period: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        if hidden_dim == 0:
            return
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be even.")
        half_dim = hidden_dim // 2
        div_term_exponent = torch.arange(0, half_dim, dtype=torch.float) * (-math.log(max_period) / (half_dim - 1e-6))
        self.register_buffer('div_term', torch.exp(div_term_exponent))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        if self.hidden_dim == 0:
            return torch.empty(dt.size(0), 0, device=dt.device)
        if dt.ndim == 1:
            dt = dt.unsqueeze(1)
        sin_inp = dt * self.div_term.unsqueeze(0)
        emb = torch.zeros(dt.size(0), self.hidden_dim, device=dt.device)
        emb[:, 0::2] = torch.sin(sin_inp)
        emb[:, 1::2] = torch.cos(sin_inp)
        return emb

class HGATSolver(nn.Module):
    def __init__(self,
                 fluid_input_dim: int,
                 solid_elastic_input_dim: int,
                 fluid_output_dim: int,
                 solid_elastic_output_dim: int,
                 phys_params_dim: int,
                 coord_dim: int = 2,
                 time_emb_dim: int = 32,
                 hidden_dim: int = 128,
                 heads: int = 4,
                 layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        
        self.time_encoder = SinusoidalTimeEncoding(time_emb_dim)
        
        self.input_encoders = nn.ModuleDict({
            'fluid': Linear(fluid_input_dim + coord_dim + time_emb_dim, hidden_dim),
            'solid_elastic': Linear(solid_elastic_input_dim + coord_dim + time_emb_dim, hidden_dim),
            'solid_rigid': Linear(coord_dim + time_emb_dim, hidden_dim)
        })
        
        self.dynamic_node_types = ['fluid', 'solid_elastic']
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.aggregation_weights = nn.ModuleList()
        
        gat_out_channels = hidden_dim // heads
        
        for _ in range(layers):
            conv_dict = {
                'fluid__f2f__fluid': GATv2Conv(hidden_dim, gat_out_channels, heads, dropout=dropout, add_self_loops=True),
                'solid_elastic__se2se__solid_elastic': GATv2Conv(hidden_dim, gat_out_channels, heads, dropout=dropout, add_self_loops=True),
                'solid_elastic__se2f__fluid': GATv2Conv((hidden_dim, hidden_dim), gat_out_channels, heads, dropout=dropout),
                'fluid__f2se__solid_elastic': GATv2Conv((hidden_dim, hidden_dim), gat_out_channels, heads, dropout=dropout),
                'solid_rigid__sr2f__fluid': GATv2Conv((hidden_dim, hidden_dim), gat_out_channels, heads, dropout=dropout),
                'solid_rigid__sr2se__solid_elastic': GATv2Conv((hidden_dim, hidden_dim), gat_out_channels, heads, dropout=dropout),
            }
            self.conv_layers.append(nn.ModuleDict(conv_dict))
            self.norm_layers.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.dynamic_node_types}))
            self.aggregation_weights.append(nn.ParameterDict({
                'fluid_self': nn.Parameter(torch.tensor(1.0)), 'fluid_cross': nn.Parameter(torch.tensor(1.0)),
                'solid_elastic_self': nn.Parameter(torch.tensor(1.0)), 'solid_elastic_cross': nn.Parameter(torch.tensor(1.0)),
            }))

        self.gated_aggregator = nn.ModuleDict({
            nt: nn.Sequential(Linear(hidden_dim * 2 + phys_params_dim, hidden_dim), nn.Sigmoid())
            for nt in self.dynamic_node_types
        })
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_decoders = nn.ModuleDict({
            nt: Linear(hidden_dim, fluid_output_dim if nt == 'fluid' else solid_elastic_output_dim)
            for nt in self.dynamic_node_types
        })

    def forward(self, data: HeteroData):
        h_input_dict = {}
        for node_type, encoder in self.input_encoders.items():
            if data[node_type].num_nodes > 0:
                node_to_graph_idx = data[node_type].batch
                node_dt = data.dt[node_to_graph_idx]
                time_embedding = self.time_encoder(node_dt)
                if node_type in self.dynamic_node_types:
                    input_tensors = [data[node_type].x, data[node_type].pos, time_embedding]
                else:
                    input_tensors = [data[node_type].pos, time_embedding]
                combined_input = torch.cat(input_tensors, dim=-1)
                h_input_dict[node_type] = encoder(combined_input)
        
        h_dict = {nt: h.clone() for nt, h in h_input_dict.items()}

        for i in range(len(self.conv_layers)):
            h_norm_dict = {nt: self.norm_layers[i][nt](h) for nt, h in h_dict.items() if nt in self.dynamic_node_types}
            if 'solid_rigid' in h_dict:
                h_norm_dict['solid_rigid'] = h_dict['solid_rigid']

            raw_messages = {nt: {'self': [], 'cross': []} for nt in self.dynamic_node_types}

            for edge_type_str, conv_op in self.conv_layers[i].items():
                src_type, rel_type, dst_type = edge_type_str.split('__')
                if h_norm_dict.get(src_type) is None or dst_type == 'solid_rigid':
                    continue
                x_src, x_dst = h_norm_dict[src_type], h_norm_dict[dst_type]
                edge_index = data.edge_index_dict.get((src_type, rel_type, dst_type))
                if edge_index is None or edge_index.numel() == 0:
                    continue
                out = conv_op(x=(x_src, x_dst), edge_index=edge_index)
                msg_key = 'self' if src_type == dst_type else 'cross'
                raw_messages[dst_type][msg_key].append(out)
            
            weights = self.aggregation_weights[i]
            for node_type in self.dynamic_node_types:
                if data[node_type].num_nodes == 0:
                    continue
                self_msg = torch.stack(raw_messages[node_type]['self'], dim=0).sum(dim=0) if raw_messages[node_type]['self'] else torch.zeros_like(h_dict[node_type])
                cross_msg = torch.stack(raw_messages[node_type]['cross'], dim=0).sum(dim=0) if raw_messages[node_type]['cross'] else torch.zeros_like(h_dict[node_type])
                w_self_key = f'{node_type}_self'
                w_cross_key = f'{node_type}_cross'
                w_self, w_cross = weights[w_self_key], weights[w_cross_key]
                aggregated_message = w_self * self_msg + w_cross * cross_msg
                h_dict[node_type] = h_dict[node_type] + torch.relu(aggregated_message)
        
        h_final_dict = {}
        for node_type, h_gnn in h_dict.items():
            if node_type in self.gated_aggregator:
                h_initial = h_input_dict[node_type]
                phys_params = data[node_type].phys_params
                gate_input = torch.cat([h_initial, h_gnn, phys_params], dim=-1)
                gate = self.gated_aggregator[node_type](gate_input)
                h_final_dict[node_type] = (1 - gate) * h_initial + gate * h_gnn

        predictions = {}
        for nt, dec in self.output_decoders.items():
            if nt in h_final_dict:
                predictions[nt] = dec(self.dropout(h_final_dict[nt]))
        return predictions