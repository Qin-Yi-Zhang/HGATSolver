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
                 solid_input_dim: int,
                 fluid_output_dim: int,
                 solid_output_dim: int,
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
            'solid': Linear(solid_input_dim + coord_dim + time_emb_dim, hidden_dim)
        })

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.aggregation_weights = nn.ModuleList()
        
        for _ in range(layers):
            conv_dict = {
                'fluid__f2f__fluid': GATv2Conv(hidden_dim, hidden_dim // heads, heads, dropout=dropout, add_self_loops=True),
                'solid__s2s__solid': GATv2Conv(hidden_dim, hidden_dim // heads, heads, dropout=dropout, add_self_loops=True),
                'solid__interface__fluid': GATv2Conv((hidden_dim, hidden_dim), hidden_dim // heads, heads, dropout=dropout),
                'fluid__rev_interface__solid': GATv2Conv((hidden_dim, hidden_dim), hidden_dim // heads, heads, dropout=dropout),
            }
            self.conv_layers.append(nn.ModuleDict(conv_dict))
            self.norm_layers.append(nn.ModuleDict({'fluid': nn.LayerNorm(hidden_dim), 'solid': nn.LayerNorm(hidden_dim)}))
            self.aggregation_weights.append(nn.ParameterDict({
                'fluid_self': nn.Parameter(torch.tensor(1.0)),
                'fluid_cross': nn.Parameter(torch.tensor(1.0)),
                'solid_self': nn.Parameter(torch.tensor(1.0)),
                'solid_cross': nn.Parameter(torch.tensor(1.0)),
            }))

        self.gated_aggregator = nn.ModuleDict({
            'fluid': nn.Sequential(Linear(hidden_dim * 2 + phys_params_dim, hidden_dim), nn.Sigmoid()),
            'solid': nn.Sequential(Linear(hidden_dim * 2 + phys_params_dim, hidden_dim), nn.Sigmoid())
        })
          
        self.dropout = nn.Dropout(dropout)
        self.output_decoders = nn.ModuleDict({
            'fluid': Linear(hidden_dim, fluid_output_dim),
            'solid': Linear(hidden_dim, solid_output_dim)
        })

    def forward(self, data: HeteroData):
        h_input_dict = {}
        for node_type, encoder in self.input_encoders.items():
            if data[node_type].num_nodes > 0:
                node_to_graph_idx = data[node_type].batch
                node_dt = data.dt[node_to_graph_idx]
                time_embedding = self.time_encoder(node_dt)
                combined_input = torch.cat([data[node_type].x, data[node_type].pos, time_embedding], dim=-1)
                h_input_dict[node_type] = encoder(combined_input)
        
        h_dict = {nt: h.clone() for nt, h in h_input_dict.items()}

        for i in range(len(self.conv_layers)):
            h_norm_dict = {nt: self.norm_layers[i][nt](h) for nt, h in h_dict.items()}
            raw_messages = {nt: {'self': [], 'cross': []} for nt in h_dict.keys()}
            for edge_type_str, conv_op in self.conv_layers[i].items():
                src_type, rel_type, dst_type = edge_type_str.split('__')
                if h_norm_dict.get(src_type) is None:
                    continue
                x_src, x_dst = h_norm_dict[src_type], h_norm_dict[dst_type]
                edge_index = data.edge_index_dict.get((src_type, rel_type, dst_type))
                if edge_index is None or edge_index.numel() == 0:
                    continue
                out = conv_op(x=(x_src, x_dst) if src_type != dst_type else x_src, edge_index=edge_index)
                msg_key = 'self' if src_type == dst_type else 'cross'
                raw_messages[dst_type][msg_key].append(out)
            
            weights = self.aggregation_weights[i]
            for node_type, h in h_dict.items():
                self_msg = torch.stack(raw_messages[node_type]['self'], dim=0).sum(dim=0) if raw_messages[node_type]['self'] else torch.zeros_like(h)
                cross_msg = torch.stack(raw_messages[node_type]['cross'], dim=0).sum(dim=0) if raw_messages[node_type]['cross'] else torch.zeros_like(h)
                w_self, w_cross = weights[f'{node_type}_self'], weights[f'{node_type}_cross']
                aggregated_message = w_self * self_msg + w_cross * cross_msg
                h_dict[node_type] = h + torch.relu(aggregated_message)
        
        h_final_dict = {}
        for node_type, h_gnn in h_dict.items():
            if node_type in h_input_dict:
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