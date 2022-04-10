import torch 
import torch.nn as nn
import torch.nn.functional as F


import layers
GCN = layers.GCN

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,node_nums,args):
        super(Classifier, self).__init__()
    
        self.conv1 = GCN(in_dim, hidden_dim)
        self.conv2 = GCN(hidden_dim, hidden_dim)

        self.norm1 = nn.BatchNorm1d(node_nums)
        self.norm2 = nn.BatchNorm1d(node_nums)


        self.classify1 = nn.Linear(hidden_dim, n_classes)
        self.classify2 = nn.Linear(hidden_dim, n_classes)

    def forward(self,adj,h):

        h = torch.unsqueeze(h,2)
        
        h1 = F.relu(self.conv1(h, adj))
        h1 = self.norm1(h1)

        h2 = F.relu(self.conv2(h1, adj))
        h2 = self.norm2(h2)

        return torch.sigmoid(
                self.classify1(torch.mean(h1,dim=1)) 
                +  self.classify2(torch.mean(h2,dim=1))
            )
