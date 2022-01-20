import torch, math
import torch.nn as nn
import torch.nn.functional as F

class DeepSet(nn.Module):
    def __init__(self, in_channels, classes, embed=512, pool='mean'):
        super(DeepSet, self).__init__()
        

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=embed, out_channels=embed, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(embed),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed, out_channels=1, kernel_size=1, stride=1, padding=0),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=classes),
        )
        self.pool = pool
        print(f'POOLING {self.pool}')
        self.mode = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x [N,K,C,H,D]
        # since we used twostageavg that has no avgpool in encoder.
        bs = x.shape[0]
        k  = x.shape[1]
        x  = x.view(bs*k,x.shape[2],x.shape[3],x.shape[4])
        
        x = self.enc(x)
        x = x.view(bs, k,-1)

        out = []
        for i in range(bs):
            y = x[i]
            y = y.unsqueeze(0)

            if self.pool == "max":
                M = y.max(dim=1)[0]
            elif self.pool == "mean":
                M = y.mean(dim=1)
            elif self.pool == "sum":
                M = y.sum(dim=1)
            
            out.append(M.squeeze(0))

        out = torch.stack(out)
        
        if self.mode == 2:
            return out

        x = self.dec(out)
        
        return x
    
    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.cross_entropy(logits, labels)
