import torch
import torch.nn as nn
import torch.nn.functional as F

class Mutual_Attention(nn.Module):
    def __init__(self,n_features):
        super(Mutual_Attention, self).__init__()

        self.hidd_dim = n_features
        self.inter_channels = self.hidd_dim //4
        self.R_q = nn.Parameter(torch.zeros(n_features, n_features))
        self.R_k = nn.Parameter(torch.zeros(n_features, n_features))
        self.R_v = nn.Parameter(torch.zeros(n_features, n_features))
        self.R_W = nn.Parameter(torch.zeros(n_features,n_features))
        nn.init.xavier_uniform_(self.R_q)
        nn.init.xavier_uniform_(self.R_k)
        nn.init.xavier_uniform_(self.R_v)
        nn.init.xavier_uniform_(self.R_W)

        self.F_q = nn.Parameter(torch.zeros(n_features, n_features))
        self.F_k = nn.Parameter(torch.zeros(n_features, n_features))
        self.F_v = nn.Parameter(torch.zeros(n_features, n_features))
        self.F_W = nn.Parameter(torch.zeros(n_features, n_features))
        nn.init.xavier_uniform_(self.F_q)
        nn.init.xavier_uniform_(self.F_k)
        nn.init.xavier_uniform_(self.F_v)
        nn.init.xavier_uniform_(self.F_W)

        self.affinityAttConv = nn.Sequential(
            nn.Conv2d(in_channels=self.hidd_dim //2, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )

    def forward(self,head,tail,flag):
        if flag:
            bs,ch,hei,wei = head.size()
            affinityAtt = F.softmax(self.affinityAttConv(torch.cat([head,tail],dim=1)))
            alpha = affinityAtt[:,0,:,:].reshape([bs,hei*wei,1])
            alpha = alpha.expand([bs,hei * wei , hei * wei])
            batch_size = head.size(0)
            h_q = (head @ self.R_q).view(batch_size, self.inter_channels, -1)
            h_q = h_q.permute(0, 2, 1)

            t_k = (tail @ self.F_k).view(batch_size, self.inter_channels, -1)
            t_k = t_k.permute(0, 2, 1)
            t_v = (tail @ self.F_v).view(batch_size, self.inter_channels, -1)
            f_t = torch.matmul(t_k,t_v)

            h_k = (head @ self.R_k).view(batch_size, self.inter_channels, -1)
            h_k = h_k.permute(0, 2, 1)
            h_v = (head @ self.R_v).view(batch_size, self.inter_channels, -1)
            f_h = torch.matmul(h_k, h_v)

            f_div_C = F.softmax(alpha * f_t + f_h,dim=-1)

            y = torch.matmul(f_div_C,h_q)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size,self.inter_channels,*head.size()[2:])
            W_y = y @ self.R_W
            z = W_y + head
            return z
        else:
            bs, ch, hei, wei = head.size()
            affinityAtt = F.softmax(self.affinityAttConv(torch.cat([head, tail], dim=1)))
            alpha = affinityAtt[:, 1, :, :].reshape([bs, hei * wei, 1])
            alpha = alpha.expand([bs, hei * wei, hei * wei])
            batch_size = tail.size(0)
            t_q = (tail @ self.F_q).view(batch_size, self.inter_channels, -1)
            t_q = t_q.permute(0, 2, 1)

            h_k = (head @ self.R_k).view(batch_size, self.inter_channels, -1)
            h_k = h_k.permute(0, 2, 1)
            h_v = (head @ self.R_v).view(batch_size, self.inter_channels, -1)
            f_h= torch.matmul(h_k, h_v)

            t_k = (tail @ self.F_k).view(batch_size, self.inter_channels, -1)
            t_k = t_k.permute(0, 2, 1)
            t_v = (tail @ self.F_v).view(batch_size, self.inter_channels, -1)
            f_t = torch.matmul(t_k, t_v)
            f_div_C = F.softmax(alpha * f_h + f_t, dim=-1)

            y = torch.matmul(f_div_C, t_q)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *head.size()[2:])
            W_y = y @ self.F_W
            z = W_y + tail
            return z