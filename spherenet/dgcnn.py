# Modified from https://github.com/AnTao97/dgcnn.pytorch/blob/master/main_semseg.py

import torch
import torch.nn as nn

class DGCNNFeat(nn.Module):
    def __init__(self, global_feat=True, emb_dims=256):
        super(DGCNNFeat, self).__init__()
        self.global_feat = global_feat
        self.emb_dims = emb_dims  
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)

        self.fc = nn.Linear(256, self.emb_dims)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # BxDxHxW -> Bx1xDxHxW

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

        if self.global_feat:
            x = torch.max(x3, dim=2)[0]  # Global max pooling along depth dimension
            x = torch.max(x, dim=2)[0]  # Global max pooling along height dimension
            x = torch.max(x, dim=2)[0]  # Global max pooling along width dimension

            x = self.fc(x)
        else:
            x = torch.cat((x1, x2, x3), dim=1)  # Concatenate along channel dimension

        return x