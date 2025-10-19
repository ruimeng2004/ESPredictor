import torch
import torch.nn as nn
import torch.nn.functional as F
from Suplement.Network.Cap_net import CapsuleNetwork, MarginLoss
import math
    
class DrugInteractionModel(nn.Module):
    def __init__(self, feature_config, hidden_dim=64, num_classes=2, temperature=0.5):
        """
        feature_config: 例如 {
            '1D': {'dim': 128, 'start': 0},
            '2D': {'dim': 128, 'start': 128},
            '3D': {'dim': 128, 'start': 256},
            'bert': {'dim': 300, 'start': 384}
        }
        """
        super().__init__()
        self.feature_config = feature_config
        self.view_dims = {k: v['dim'] for k,v in feature_config.items()}
        
        self.view_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(config['dim'], hidden_dim),
                nn.ReLU()
            ) for name, config in feature_config.items()
        })
        
        self.cross_attention = CrossAttention(
            view_dims=self.view_dims,
            hidden_dim=hidden_dim
        )
        
        self.capsnet = CapsuleNetwork(
            input_dim=hidden_dim * len(self.view_dims),
            num_classes=num_classes,
            temperature=temperature
        )
        self.margin_loss = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5)
    
    def split_features(self, x):

        return {
            name: x[:, :, config['start']:config['start']+config['dim']]
            for name, config in self.feature_config.items()
        }
    
    def forward(self, drug1, drug2):
        probs, _ = self._forward(drug1, drug2)
        return probs
    
    def _forward(self, drug1, drug2):

        drug1_views = self.split_features(drug1.unsqueeze(1))  
        drug2_views = self.split_features(drug2.unsqueeze(1))
        
        encoded_A = {k: self.view_encoders[k](v.squeeze(1)) for k,v in drug1_views.items()}
        encoded_B = {k: self.view_encoders[k](v.squeeze(1)) for k,v in drug2_views.items()}
        
        fused_A = self.cross_attention(encoded_A, encoded_B)
        fused_B = self.cross_attention(encoded_B, encoded_A)
        
        combined_A = torch.cat([fused_A[name] for name in self.view_dims], dim=1)
        combined_B = torch.cat([fused_B[name] for name in self.view_dims], dim=1)
        
        combined = (combined_A + combined_B) / 2
        
        probs, digit_caps = self.capsnet(combined)
        return probs, digit_caps

class CrossAttention(nn.Module):
    def __init__(self, view_dims, hidden_dim=64):
        super().__init__()
        self.view_names = list(view_dims.keys())
        self.hidden_dim = hidden_dim 
        
        self.query_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim)  
            for name in view_dims.keys()
        })
        self.key_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim) 
            for name in view_dims.keys()
        })
        self.value_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim) 
            for name in view_dims.keys()
        })
        
    def forward(self, drugA_views, drugB_views):
        
        fused_views = {name: [] for name in self.view_names}
        
        for name_A in self.view_names:
            for name_B in self.view_names:
                Q = self.query_proj[name_A](drugA_views[name_A])
                K = self.key_proj[name_B](drugB_views[name_B])
                V = self.value_proj[name_B](drugB_views[name_B])
                
                attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(Q.size(-1))
                attn_weights = F.softmax(attn_scores, dim=-1)
                fused = torch.matmul(attn_weights, V)
                
                fused_views[name_A].append(fused)
        
        final_views = {}
        for name in self.view_names:

            all_fusions = torch.stack(fused_views[name], dim=1)  # [batch, num_views, hidden]
            final_views[name] = torch.mean(all_fusions, dim=1)
            
        return final_views