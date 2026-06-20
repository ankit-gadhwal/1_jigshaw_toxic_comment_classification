import torch
import torch.nn as nn
import torch.nn.functional as F
class BiLSTM_Attention(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,n_layers,dropout):
        super(BiLSTM_Attention,self).__init__()
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,n_layers,
        batch_first=True,bidirectional=True,dropout=dropout)
        self.attention_layer = nn.Linear(hidden_dim * 2,hidden_dim * 2)
        self.context_vector = nn.Linear(hidden_dim * 2,1,bias=False)
        self.fc = nn.Linear(hidden_dim * 2,1)
        self.dropout = nn.Dropout(dropout)
    def attention(self,lstm_output):
        u = torch.tanh(self.attention_layer(lstm_output))
        a = F.softmax(self.context_vector(u),dim=1)
        scored_output = torch.sum(a * lstm_output,dim=1)
        return scored_output
    def forward(self,x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        out = self.dropout(attn_out)
        out = self.fc(out).squeeze(1)
        return torch.sigmoid(out)