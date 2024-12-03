# Modified from https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_semseg.py

import torch
import torch.nn as nn

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    # device = torch.device('cpu')
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature      # (batch_size, 2*num_dims, num_points, k)

class DGCNNFeat(nn.Module):
    def __init__(self, global_feat=True, emb_dims=512):
        super(DGCNNFeat, self).__init__()
        self.global_feat = global_feat
        self.emb_dims = emb_dims  # Define the emb_dims attribute
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # Add channel dimension: Bx1xDxHxW

        x = self.conv1(x)  # Bx1xDxHxW -> Bx64xDxHxW
        x = self.bn1(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)

        x = self.conv2(x)  # Bx64xDxHxW -> Bx64xDxHxW
        x = self.bn2(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x1 = x

        x = self.conv3(x)  # Bx64xDxHxW -> Bx128xDxHxW
        x = self.bn3(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x2 = x

        x = self.conv4(x)  # Bx128xDxHxW -> Bx256xDxHxW
        x = self.bn4(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x3 = x

        x = self.conv5(x)  # Bx256xDxHxW -> Bx512xDxHxW
        x = self.bn5(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x4 = x

        if self.global_feat:
            x = torch.max(x4, dim=2)[0]  # Global max pooling along depth dimension
            x = torch.max(x, dim=2)[0]  # Global max pooling along height dimension
            x = torch.max(x, dim=2)[0]  # Global max pooling along width dimension
        else:
            x = torch.cat((x1, x2, x3, x4), dim=1)  # Concatenate along channel dimension

        return x