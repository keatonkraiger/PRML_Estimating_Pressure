import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

class MultiLevelSelfAttentionTransformer(nn.Module):
    def __init__(self, transformer_dim, num_heads, num_layers, seq_len, dropout_p=0.1, 
                 pos='learnable', mlp_dim=1024, activate='gelu', temporal_window=2):
        super().__init__()
        self.pos = pos
        
        # Setup positional encodings
        if pos == 'learnable':
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, transformer_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
            
        # Create transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attention': FrameAttentionBlock(
                    dim=transformer_dim, 
                    num_heads=num_heads,
                    temporal_window=temporal_window
                ),
                'feed_forward': nn.Sequential(
                    nn.Linear(transformer_dim, mlp_dim),
                    nn.GELU() if activate == 'gelu' else nn.ReLU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(mlp_dim, transformer_dim)
                )
            }) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(transformer_dim)
        
    def forward(self, x):
        # Add positional encodings
        if self.pos in ['learnable']:
            x = x + self.positional_encodings(x)

            
        # Pass through transformer layers
        for layer in self.layers:
            # Apply cross attention
            attended = layer['cross_attention'](x)
            # Apply feed-forward network with residual connection
            x = attended + layer['feed_forward'](attended)
            
        return self.norm(x)

class FrameAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, temporal_window=2):
        super().__init__()
        self.num_heads = num_heads
        self.intra_frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.inter_frame_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_window = temporal_window
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        
        # Intra-frame attention (local attention within each frame)
        x_intra = self.intra_frame_attn(x, x, x)[0]
        x = self.norm1(x + x_intra)
        
        # Inter-frame attention (global attention between frames with temporal masking)
        temporal_mask = self.create_temporal_mask(seq_len, self.temporal_window)
        temporal_mask = temporal_mask.to(x.device)
        
        # Expand mask for multi-head attention:
        # 1. Add batch dimension: (1, seq_len, seq_len)
        # 2. Expand to match number of heads: (batch_size * num_heads, seq_len, seq_len)
        expanded_mask = temporal_mask.unsqueeze(0).expand(batch_size * self.num_heads, -1, -1)
        
        # Apply inter-frame attention with temporal masking
        x_inter = self.inter_frame_attn(x, x, x, attn_mask=expanded_mask)[0]
        x = self.norm2(x + x_inter)
        
        return x
    
    def create_temporal_mask(self, seq_len, window):
        """
        Create a temporal mask that allows attention only within a local window.
        
        Args:
            seq_len (int): Length of the sequence
            window (int): Size of the temporal window (one-sided)
            
        Returns:
            torch.Tensor: Mask tensor of shape (seq_len, seq_len) where 0 indicates
                         allowed attention and -inf blocks attention
        """
        # Start with all -inf (block all attention)
        mask = torch.full((seq_len, seq_len), float('-inf'))
        
        # For each position, allow attention to nearby frames within the window
        for i in range(seq_len):
            # Calculate valid range for attention
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            mask[i, start:end] = 0.0  # Allow attention within window
            
        return mask
    
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py#L9
    """

    def __init__(self, in_features, out_features, bias=True, node_n=13):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PoseEmbedder(nn.Module):
    def __init__(self, num_joints, joint_dim, pose_embed_dim, seq_len=13, pose_embedder='linear'):
        super().__init__()
        self.pose_embedder = pose_embedder
        input_dim = num_joints * joint_dim
        
        if pose_embedder == 'gcn':
            self.embedding = GraphConvolution(
                in_features=input_dim,
                out_features=pose_embed_dim,
                node_n=seq_len,
                bias=True
            )
        elif pose_embedder == 'linear':
            self.embedding = nn.Linear(input_dim, pose_embed_dim)
            
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_joints, joint_dim)
        """
        batch_size, seq_len, num_joints, joint_dim = x.shape
    
        if self.pose_embedder == 'gcn':
            x = x.reshape(batch_size, seq_len, num_joints * joint_dim) # -> [batch_size, seq_len, num_joints * joint_dim]
            x = self.embedding(x)  # [batch_size, seq_len, pose_embed_dim]
        
        elif self.pose_embedder == 'linear':
            x = x.reshape(batch_size, seq_len, num_joints * joint_dim)
            x = self.embedding(x)  # [batch_size, seq_len, pose_embed_dim]
            
        else:  # temporal conv
            x = x.permute(0, 2, 3, 1).reshape(batch_size, joint_dim * num_joints, seq_len)
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
        return x
     
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class ContactConditionedPressureHead(nn.Module):
    """
    Pressure prediction head that is conditioned on contact predictions through cross-attention.
    """
    def __init__(self, input_dim, hidden_dim, pressure_dim, contact_dim, pred_distribution=False):
        super().__init__()
        
        self.contact_projection = nn.Linear(contact_dim, hidden_dim)
        self.pressure_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Residual FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.pressure_gate = nn.Sequential(
            nn.Linear(hidden_dim, pressure_dim),
            nn.Sigmoid()
        )
        
        # Final pressure prediction
        self.pressure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, pressure_dim),
        )

    def forward(self, x, contact_pred):
        # Project inputs
        pressure_query = self.pressure_projection(x).unsqueeze(1)
        contact_key = self.contact_projection(contact_pred).unsqueeze(1)
        
        # Cross attention between pressure and contact features
        attended, _ = self.cross_attention(
            pressure_query, contact_key, contact_key
        )
        
        # First residual connection
        x = self.layer_norm1(pressure_query + attended)
        
        # FFN block
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        x = x.squeeze(1)
        
        # Generate predictions with gating
        gate = self.pressure_gate(x)
        pressure = self.pressure_head(x)
        
        # Apply contact-based gating to pressure
        gated_pressure = F.log_softmax(pressure * gate, dim=-1)
       
        return torch.exp(gated_pressure)
     
class FootFormer(nn.Module):
    def __init__(self, num_joints, joint_dim, pose_embed_dim, num_heads, num_layers, output_dims, seq_len=5, dropout_p=0.1, transformer='transformer', 
                 pos='learnable', mlp_dim=1024, pool='attn',  decoder_dim=1024, mode='pressure', pose_embedder='gcn', pred_distribution=False, contact_conditioned=True):
        super().__init__()
        self.sequence_length = seq_len
        self.pool = pool
        self.mode = mode 
        self.pred_distribution = pred_distribution
       
        self.input_norm = nn.LayerNorm(joint_dim * num_joints)  
        self.pose_embedder = PoseEmbedder(
            num_joints=num_joints, 
            joint_dim=joint_dim, 
            pose_embed_dim=pose_embed_dim, 
            pose_embedder=pose_embedder,
            seq_len=seq_len
        ) 
        self.dropout = nn.Dropout(dropout_p) 
        self.pre_encoder_norm = nn.LayerNorm(pose_embed_dim)
        if transformer == 'multi':
            self.transformer = MultiLevelSelfAttentionTransformer(
                transformer_dim=pose_embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=seq_len,
                dropout_p=dropout_p,
                pos=pos,
                mlp_dim=mlp_dim
            )
 
        self.norm = nn.LayerNorm(pose_embed_dim)
        
        if pool == 'weighted':
            self.global_pool = nn.AdaptiveAvgPool1d(1)
        elif pool == 'attn':
            self.attention_pool = nn.Linear(pose_embed_dim, 1)
        else:
            pool = nn.Identity()
    
        # Create task-specific heads
        self.task_heads = nn.ModuleDict()
       
        self.output_norms = nn.ModuleDict({
            task: nn.LayerNorm(pose_embed_dim) for task in mode
        })
             
        task_activations = {
            'pressure': nn.Softmax(dim=-1) if pred_distribution else nn.Identity(),  
            'contact': nn.Identity(),    # Sigmoid for binary contact
            'com': nn.Identity()        
        }
         
        # Create heads for each task
        self.task_heads['pressure'] = ContactConditionedPressureHead(
            input_dim=pose_embed_dim,
            hidden_dim=decoder_dim,
            pressure_dim=output_dims['pressure'],
            contact_dim=output_dims['contact']
        )

            
        self.task_heads['contact'] = self._create_head(
            pose_embed_dim,
            decoder_dim,
            output_dims['contact'],
            task_activations['contact']
        )
        
        self.task_heads['com'] = self._create_head(
            pose_embed_dim,
            decoder_dim,
            output_dims['com'],
            task_activations['com']
        )
        
    def _create_head(self, input_dim, hidden_dim, output_dim, out_activation=nn.Identity()):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim),  
            nn.Linear(hidden_dim, output_dim),
            out_activation
        )
         
    def apply_pool(self, x):
        attn_weights = self.attention_pool(x)  # [batch, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        # Add dropout to attention weights
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        x = torch.bmm(attn_weights.transpose(1,2), x).squeeze(1)
        return x
            
    def forward(self, x):
        """Expect input x to be of shape (batch_size, sequence_length, num_joints, in_channels)"""
        # Embedd pose sequence
        batch_size, seq_len, num_joints, joint_dim = x.shape
        x = x.reshape(batch_size, seq_len, -1)
        x = self.input_norm(x)
        x = x.reshape(batch_size, seq_len, num_joints, joint_dim) 
        
        x = self.pose_embedder(x) 
        x = self.dropout(x)
        x = self.pre_encoder_norm(x)
           
        # Pass through transformer 
        x = self.transformer(x)
        x = self.norm(x)
        
        # Apply pooling
        x = self.apply_pool(x)

        outputs = {}
       
        contact_norm = self.output_norms['contact'](x)
        outputs['contact'] = self.task_heads['contact'](contact_norm)
       
        # Handle all tasks
        for task in self.mode:
            if task not in outputs:  # Skip contact if already computed
                task_norm = self.output_norms[task](x)
                if task == 'pressure' and self.contact_conditioned:
                    outputs[task] = self.task_heads[task](task_norm, outputs['contact'])
                else:
                    out = self.task_heads[task](task_norm)
                    if task == 'com':
                        out[..., -1] = F.relu(out[..., -1])
                    outputs[task] = out
        
        return outputs
    