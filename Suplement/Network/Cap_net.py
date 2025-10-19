import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, input_dim, num_capsules=8, caps_dim=8):
        super().__init__()
        self.conv1d = nn.Conv1d(1, num_capsules*caps_dim, kernel_size=3, stride=1, padding=1)
        self.num_capsules = num_capsules
        self.caps_dim = caps_dim
        
    def squash(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))
        
    def forward(self, x):
        x = x.unsqueeze(1)  
        out = self.conv1d(x)  
        out = out.reshape(out.size(0), self.num_capsules, self.caps_dim, -1)  
        out = out.transpose(2, 3) 
        return self.squash(out)

class AttentionAggregation(nn.Module):
    def __init__(self, num_capsules, caps_dim):
        super().__init__()
        self.attn_layer = nn.Linear(caps_dim, 1) 
        
    def forward(self, caps_out):
        attn_weights = torch.softmax(
            self.attn_layer(caps_out).squeeze(-1), 
            dim=-1
        )  
        return (attn_weights.unsqueeze(-1) * caps_out).sum(dim=2) 

class DigitCapsuleLayer(nn.Module):
    def __init__(self, in_capsules=8, in_dim=8, out_capsules=2, out_dim=16, routing_iters=3):
        super().__init__()
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, out_dim, in_dim))
        
    def squash(self, s):
        norm = torch.norm(s, dim=-1, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))
        
    def forward(self, x):
        batch_size = x.size(0)
        u_hat = torch.matmul(self.W, x.unsqueeze(2).unsqueeze(4)).squeeze(-1)
        
        with torch.autograd.set_grad_enabled(True):
            b = torch.zeros(batch_size, x.size(1), u_hat.size(2), 1).to(x.device)
            for i in range(self.routing_iters):
                c = F.softmax(b, dim=2)
                s = (c * u_hat).sum(dim=1, keepdim=True)
                v = self.squash(s)
                if i != self.routing_iters - 1:
                    b = b + (u_hat * v).sum(dim=-1, keepdim=True)
                    
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
        return v.squeeze() 
    
class CapsuleNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=2, temperature=1.0):
        super().__init__()
        self.temperature = temperature  
        
        self.primary_caps = PrimaryCapsuleLayer(
            input_dim=input_dim,
            num_capsules=8,
            caps_dim=8
        )
        
        self.attention_agg = AttentionAggregation(
            num_capsules=8,
            caps_dim=8
        )
        
        self.digit_caps = DigitCapsuleLayer(
            in_capsules=8,
            in_dim=8,
            out_capsules=num_classes,
            out_dim=16,
            routing_iters=3
        )
        
    def forward(self, x):
        primary_out = self.primary_caps(x) 
        caps_agg = self.attention_agg(primary_out) 
        digit_out = self.digit_caps(caps_agg)  
        caps_length = torch.norm(digit_out, dim=-1)
        probs = F.softmax(caps_length / self.temperature, dim=-1)  
        
        return probs, digit_out 
    
class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        
    def forward(self, v, labels):
        v_norm = torch.norm(v, dim=-1) 
        loss = labels * F.relu(self.m_plus - v_norm).pow(2) + \
               self.lambda_ * (1 - labels) * F.relu(v_norm - self.m_minus).pow(2)
        
        return loss.mean()