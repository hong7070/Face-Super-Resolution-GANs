import argparse
import os
from math import log10
from torch import log10 as tlog10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss_identity import GeneratorLoss
from model import Generator, Discriminator

from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=8, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


def rgb_to_y(x):
    # x: Tensor shape [..., 3, H, W], values in [0,1]
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    return 0.299*r + 0.587*g + 0.114*b

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pre‐trained on VGGFace2
face_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
# reusable cosine‐similarity
cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=1, shuffle=False)
    
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': [], 'cosine': []}
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.float().cuda()
            z = data
            if torch.cuda.is_available():
                z = z.float().cuda()
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            real_out = netD(real_img).mean()
            fake_out = netD(fake_img.detach()).mean()
            d_loss = 1 - real_out + fake_out

            optimizerD.zero_grad()
            d_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerD.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'cosine': 0, 'batch_sizes': 0}
            
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                
                lr = val_lr.float().to(device)
                hr = val_hr.float().to(device)
                # if torch.cuda.is_available():
                #     lr = lr.float().cuda()
                #     hr = hr.float().cuda()
                sr = netG(lr)
                
                # 1) Resize to the 160×160 that the embedder expects
                sr_embed = F.interpolate(sr, size=(160,160), mode='bilinear', align_corners=False)
                hr_embed = F.interpolate(hr, size=(160,160), mode='bilinear', align_corners=False)
                
                # 2) Normalize to [-1,1] from [0, 1] (facenet-pytorch convention)
                sr_embed = (sr_embed - 0.5) / 0.5
                hr_embed = (hr_embed - 0.5) / 0.5
                
                # 3) Extract embeddings
                emb_sr = face_model(sr_embed)
                emb_hr = face_model(hr_embed)
                
                # 4) Compute mean cosine similarity over the batch
                batch_cos = cosine_sim(emb_sr, emb_hr).mean().item()
                # accumulate
                valing_results['cosine']  += batch_cos * batch_size              
                
                sr_y = rgb_to_y(sr)
                hr_y = rgb_to_y(hr)
                batch_mse = ((sr_y - hr_y) ** 2).mean()                
                
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                
                # valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                # since Y in [0,1], PSNR simplifies to:
                psnr_tensor = 10 * tlog10(1.0 / (valing_results['mse'] / valing_results['batch_sizes']))
                # detach from GPU and convert to Python float
                valing_results['psnr'] = psnr_tensor.cpu().item()
                
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                
                                
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            valing_results['cosine'] /= valing_results['batch_sizes']
            results['cosine'].append(valing_results['cosine'])  
            
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
                
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
       
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim'], 'Cosine':  results['cosine']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
