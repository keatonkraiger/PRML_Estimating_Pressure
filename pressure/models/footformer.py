import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import math

from pressure.models.gcn import GraphConvolution

############# Util#############
def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def exists(val):
    return val is not None

class ScaledSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert dim % 2 == 0
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32) / half_dim 
        inv_freq = theta ** (-freq_seq)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        seq_len, device = x.shape[1], x.device
        pos = torch.arange(seq_len, device=device)
        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        pos = pos * self.scale
        if pos.ndim == 1:
            pos = pos.unsqueeze(1)
        return pos
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        """
        
        Args:
            dim_model (int): embedding dimension
            dropout_p (float): dropout probability
            max_len (int): length of sequences 
        """
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding)

############# Pose Embedder #############
class TemporalConvNet(nn.Module):
    def __init__(self, num_joints, in_channels, out_channels, kernel_size):
        """

        Args:
            num_joints (int): num joint
            in_channels (int): dimensionality of each joint  
            out_channels (int): embedding dimension 
            kernel_size (int): kernel size for temporal convolution 
        """
        super().__init__()
        self.temporal_conv = nn.Conv1d(in_channels * num_joints, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.apply(self.init_weight)
        
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
             
    def forward(self, x):
        """Expect input x to be of shape (batch_size, sequence_length, num_joints, in_channels)"""
        x = self.temporal_conv(x)
        return x
###########################################

############# CrossTransformer #############
class CrossTransformer(nn.Module):
    def __init__(self, transformer_dim, num_heads, num_layers, seq_len, dropout_p=0.1, 
                 pos='learnable', mlp_dim=1024, activate='gelu'):
        super().__init__()
        self.pos = pos
        if pos == 'learnable':
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, transformer_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == 'sinusoidal':
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(transformer_dim)
        else:
            self.positional_encodings = PositionalEncoding(transformer_dim, dropout_p, seq_len)
            
        # Create layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attention': DualAttentionBlock(transformer_dim, num_heads),
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
        # x shape: (batch_size, seq_len, transformer_dim)
        
        # Add positional encodings
        if self.pos in ['learnable', 'sinusoidal']:
            x += self.positional_encodings(x)
        else:
            x = self.positional_encodings(x)
            
        # Pass through layers
        for layer in self.layers:
            x = layer['cross_attention'](x)
            x = x + layer['feed_forward'](x)
            
        return self.norm(x)

class DualAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        
        # Self attention within sequence
        x_self = self.self_attn(x, x, x)[0]
        x = self.norm1(x + x_self)
        
        # Cross attention between neighboring frames
        x_cross = self.cross_attn(x, x, x)[0]  
        x = self.norm2(x + x_cross)
        return x
################################################  
    
############# MultiLevelSelfAttentionTransformer #############
class MultiLevelSelfAttentionTransformer(nn.Module):
    def __init__(self, transformer_dim, num_heads, num_layers, seq_len, dropout_p=0.1, 
                 pos='learnable', mlp_dim=1024, activate='gelu', temporal_window=2):
        super().__init__()
        self.pos = pos
        
        # Setup positional encodings
        if pos == 'learnable':
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, transformer_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == 'sinusoidal':
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(transformer_dim)
        else:
            self.positional_encodings = PositionalEncoding(transformer_dim, dropout_p, seq_len)
            
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
        if self.pos in ['learnable', 'sinusoidal']:
            x = x + self.positional_encodings(x)
        else:
            x = self.positional_encodings(x)
            
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
################################################

############# Transformer #############
class Transformer(nn.Module):
    def __init__(self, transformer_dim=128, num_heads=8, num_layers=6, seq_len=13, dropout_p=0.1, pos='sinusoidal', mlp_dim=1024, activate='gelu'):
        """
        Args:
            transformer_dim (int, optional): embedding dimension of transformer layers . Defaults to 128.
            num_heads (int, optional): num transformer heads. Defaults to 8.
            num_layers (int, optional): num of encoder layers. Defaults to 6.
            seq_len (int, optional): length of sequences. Defaults to 5.
            dropout_p (float, optional): dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.pos = pos
        if pos == 'learnable':
            self.positional_encodings = nn.Parameter(torch.zeros(seq_len, transformer_dim), requires_grad=True)
            nn.init.trunc_normal_(self.positional_encodings, std=0.2)
        elif pos == 'sinusoidal':
            self.positional_encodings = ScaledSinusoidalPositionalEncoding(transformer_dim)
        else:
            self.positional_encodings = PositionalEncoding(transformer_dim, dropout_p, seq_len)
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads,  batch_first=True, dropout=dropout_p, dim_feedforward=mlp_dim, activation=activate)
        layer_norm = nn.LayerNorm(transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=layer_norm)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and exists(m.bias):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
             
    def forward(self, x):
        """Expect input x to be of shape (batch_size, sequence_length, transformer_dim)"""
        if self.pos in ['learnable', 'sinusoidal']:
            x += self.positional_encodings(x)
        else:
            x = self.positional_encodings(x)
        x = self.transformer_encoder(x)
        return x
################################################

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
        else:
            self.embedding = TemporalConvNet(num_joints, joint_dim, pose_embed_dim, kernel_size=3)

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
                 pos='learnable', mlp_dim=1024, pool='attn',  decoder_dim=1024, mode='pressure', pose_embedder='gcn', pred_distribution=False, contact_conditioned=False):
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
        elif transformer == 'cross':
            self.transformer = CrossTransformer(
                transformer_dim=pose_embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                seq_len=seq_len,
                dropout_p=dropout_p,
                pos=pos,
                mlp_dim=mlp_dim
            )
        else:
            self.transformer = Transformer(
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
        self.contact_conditioned = contact_conditioned
        if 'pressure' in mode:
            self.task_heads['pressure'] = self._create_head(
                pose_embed_dim, 
                decoder_dim, 
                output_dims['pressure'],
                task_activations['pressure']
            )
            
        if 'contact' in mode:
            if contact_conditioned:
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
            
        if 'com' in mode:
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
        if self.pool == 'weighted':
            x = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)
        elif self.pool == 'attn':
            attn_weights = self.attention_pool(x)  # [batch, seq_len, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            # Add dropout to attention weights
            attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
            x = torch.bmm(attn_weights.transpose(1,2), x).squeeze(1)
        else:
            x = x[:, x.shape[1]//2, :]
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
       
        # Get contact predictions first if using conditioning
        if self.contact_conditioned and 'contact' in self.mode:
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