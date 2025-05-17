import torch
from torch import nn
from torchvision.models.vgg import vgg16

from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import Resize, ToTensor


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        
        #------- identity network ----------------------------------------------------
        self.id_net = InceptionResnetV1(pretrained='vggface2').eval()
        for param in self.id_net.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()
        #-----------------------------------------------------------------------------

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)   # L_GAN
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))   # L_content 
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)   # L_pixel
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        
        #------ compute identity loss -------------------------------------------------
        # 1) resize to 160×160 and normalize to (0,1) -> (–1,1)
        id_sr = Resize((160,160))(out_images)          # PIL‑style transform works on tensor
        id_hr = Resize((160,160))(target_images)
        # 2) FaceNet expects inputs in [–1,1]
        id_sr = (id_sr - 0.5) / 0.5
        id_hr = (id_hr - 0.5) / 0.5
        # 3) extract embeddings
        emb_sr = self.id_net(id_sr)
        emb_hr = self.id_net(id_hr)
        # 4) MSE between embeddings
        identity_loss = self.mse_loss(emb_sr, emb_hr)
        
        # ——— one‐time debug print ———
        if not hasattr(self, '_logged'):
            print(
                f" pixel={image_loss.item():.4f},"
                f" gan={adversarial_loss.item():.4f},"
                f" content={perception_loss.item():.4f},"
                f" tv={tv_loss.item():.4e},"
                f" id={identity_loss.item():.4f}"
            )
            self._logged = True
        
        # ——— total loss with identity term ———
        return image_loss + 0.001 * adversarial_loss + 0.005 * perception_loss + 2e-8 * tv_loss + 0.6 * identity_loss # tune identity weights as needed
        # return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        #-----------------------------------------------------------------------------


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
