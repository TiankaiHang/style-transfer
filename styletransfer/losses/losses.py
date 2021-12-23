import torch
from torchvision.models import vgg16
import torch.nn.functional as F

class PerceptualLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = vgg16(pretrained=True).features
        
        """
        Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace=True)
            (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace=True)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace=True)
            (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace=True)
            (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace=True)
            (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace=True)
            (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        """

    def extract_features(self, x, layer_inds=[3, 8, 15, 22]):
        """relu1_2, relu2_2, relu3_3, and relu4_3"""
        feat_list = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in layer_inds:
                feat_list.append(x)

        return feat_list

    def feat_reconstruct_loss(self, x, y):
        return F.mse_loss(x, y, reduction='mean')

    def get_gram_matrix(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        return torch.bmm(x, x.permute(0, 2, 1)) / (C * H * W)
    
    def style_reconstruct_loss(self, x, y):
        G_x = self.get_gram_matrix(x)
        G_y = self.get_gram_matrix(y)

        # return torch.norm(G_x - G_y, p="fro")
        return F.mse_loss(G_x, G_y)

    def forward(self, y_s, y_hat, y_c):
        feat_list_y_s = self.extract_features(y_s)
        feat_list_y_hat = self.extract_features(y_hat)
        feat_list_y_c = self.extract_features(y_c, layer_inds=[8])

        style_loss = 0.0

        # style loss
        for feat_y_s, feat_y_hat in zip(feat_list_y_s, feat_list_y_hat):
            style_loss += self.style_reconstruct_loss(feat_y_s, feat_y_hat)

        # feat loss
        feat_loss = self.feat_reconstruct_loss(feat_list_y_c[0], feat_list_y_hat[1])

        return style_loss, feat_loss


if __name__ == '__main__':
    loss_fn = PerceptualLoss().cuda()

    x = torch.randn(4, 3, 256, 256).cuda()
    y = torch.randn(4, 3, 256, 256).cuda()
    z = torch.randn(4, 3, 256, 256).cuda()

    loss = loss_fn(x, y, z)

    loss.backward()

    print(loss)