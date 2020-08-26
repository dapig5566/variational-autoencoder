import time
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from Loader import DataManager
from torch.distributions import MultivariateNormal
from torchvision.models import mobilenet_v2

def elbo_loss(x, encoder, decoder):
    m = x.shape[0]
    enc_param = encoder(x).view([m, 2, -1]).repeat_interleave(L, dim=0)
    loc, scale_logit = enc_param[:, 0, :], enc_param[:, 1, :]
    scale = torch.sqrt(torch.exp(scale_logit))
    term1 = 0.5 * torch.sum(1 + scale_logit - loc**2 - torch.exp(scale_logit))
    epi = torch.randn(loc.shape).cuda()
    z = (epi * scale + loc)#.unsqueeze(-1).repeat_interleave(16, dim=-1).view([-1, 128, 4, 4])#torch.cat([epi[i*L:(i+1)*L] * scale[i] + loc[i] for i in range(m)])
    dec_param = decoder(z).view([m*L, 6, -1]).transpose(0, 1)
    r_u, r_s, g_u, g_s, b_u, b_s = dec_param

    x = torch.repeat_interleave(x, L, dim=0)
    pdf_r = MultivariateNormal(r_u, torch.diag_embed(torch.exp(r_s)))
    pdf_g = MultivariateNormal(g_u, torch.diag_embed(torch.exp(g_s)))
    pdf_b = MultivariateNormal(b_u, torch.diag_embed(torch.exp(b_s)))
    term2 = pdf_r.log_prob(x[:, 0, ...].flatten(1)).sum() + pdf_g.log_prob(x[:, 1, ...].flatten(1)).sum() + pdf_b.log_prob(x[:, 2, ...].flatten(1)).sum()
    eblo = -(term1 + term2 / L)
    return eblo

def generate_images(x):
    with torch.no_grad():
        m = x.shape[0]
        enc_params = encoder(x).view([m, 2, -1])
        loc, scale_logit = enc_params[:, 0, :], enc_params[:, 1, :]
        scale = torch.sqrt(torch.exp(scale_logit))
        epi = torch.randn(loc.shape).cuda()
        z = (epi * scale + loc)#.unsqueeze(-1).repeat_interleave(16, dim=-1).view([-1, 128, 4, 4])
        mean = dataset.mean
        std = dataset.std
        params = decoder(z).view([m, 6, -1]).transpose(0, 1)
        r_u, r_s, g_u, g_s, b_u, b_s = params
        imgs = torch.cat([(r_u + torch.exp(r_s).sqrt() * torch.randn(r_u.shape).cuda()).view(m, 32, 32).unsqueeze(1),
                          (g_u + torch.exp(g_s).sqrt() * torch.randn(g_u.shape).cuda()).view(m, 32, 32).unsqueeze(1),
                          (b_u + torch.exp(b_s).sqrt() * torch.randn(b_u.shape).cuda()).view(m, 32, 32).unsqueeze(1)], dim=1).cpu().numpy()
        imgs = imgs * std + mean
        x = x.cpu().numpy() * std + mean
    fig, rows = plt.subplots(2, 4)
    for i, ax in enumerate(rows[0]):
        ax.imshow(imgs[i].transpose(1, 2, 0).astype('uint8'))
    for i, ax in enumerate(rows[1]):
        ax.imshow(x[i].transpose(1, 2, 0).astype('uint8'))
    plt.show()

L = 8
batch_size = 4
epochs = 100
iters = 300#50000 // batch_size + 1
show_every = 10
dataset = DataManager('/home/liu/WORKSPACE/DATASET', use_augment=False)
dataset.load_CIFAR_10()

#decoder = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                        # nn.ReLU(),
                        # nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
                        # nn.ReLU(),
                        # nn.ConvTranspose2d(32, 6, 3, 2, padding=1, output_padding=1)).cuda()
decoder = nn.Sequential(nn.Linear(128, 2048, bias=False),
                        nn.BatchNorm1d(2048),
                        nn.ReLU(),
                        nn.Linear(2048, 32*32*6, bias=True)).cuda()
# encoder = mobilenet_v2(num_classes=256*2).cuda()
encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.ReLU6(),
                        nn.Conv2d(32, 64, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU6(),
                        nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU6(),
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(128, 128*2)).cuda()
opt = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 0.0)
batch_x, _ = dataset.sample(batch_size)
batch_x = torch.from_numpy(batch_x).cuda()
s = time.time()
for i in range(epochs):
    for j in range(iters):

        opt.zero_grad()
        loss = elbo_loss(batch_x, encoder, decoder)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 3.5, 2)
        opt.step()
        scheduler.step(j/iters + i)
        if j % show_every == 0:
            e = time.time()
            print('ep {}, {}/{}: {:.5f}'.format(i, j, iters, loss.item()))
            print('time elapsed: {:.5f}'.format((e-s)/60))
            s = time.time()

        if j % 50 == 0:
            generate_images(batch_x)




