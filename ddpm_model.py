import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels: out = x + x2
            else: out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        )
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x).view(-1, self.emb_dim, 1, 1)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), 
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        
        # c * mask: 如果 mask 是 1，保留条件；如果 mask 是 0，变成空条件
        c = c * context_mask 
        
        cemb1 = self.contextembed1(c)
        temb1 = self.timeembed1(t)
        cemb2 = self.contextembed2(c)
        temb2 = self.timeembed2(t)
        up0 = self.up0(hiddenvec) 
        up1 = self.up1(cemb1 * up0 + temb1, down2) 
        up2 = self.up2(cemb2 * up1 + temb2, down1) 
        out = self.out(torch.cat((up2, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / (sqrtmab + 1e-4)
    return {
        "alpha_t": alpha_t, "oneover_sqrta": oneover_sqrta, "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t, "sqrtab": sqrtab, "sqrtmab": sqrtmab, "mab_over_sqrtmab": mab_over_sqrtmab,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in betas.items(): self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x) 
        x_t = self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * noise
        
        context_mask = torch.bernoulli(torch.zeros_like(c) + (1 - self.drop_prob)).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, c_labels=None, return_all_steps=False):
        x_i = torch.randn(n_sample, *size).to(device)
        
        frames = []
        if return_all_steps:
            frames.append(x_i.clone())

        if c_labels is not None:
            c_i = c_labels.to(device)
        else:
            c_i = torch.arange(0, 10).to(device)
            c_i = c_i.repeat(int(n_sample/10) + 1)
            c_i = c_i[:n_sample]
            c_i = F.one_hot(c_i, num_classes=10).float()

        # 初始化为全 1 (表示默认全是“有条件”的)
        context_mask = torch.ones_like(c_i).to(device)
        
        c_i = c_i.repeat(2, 1)
        context_mask = context_mask.repeat(2, 1)
        
        # 把后半部分置为 0 (作为 CFG 的无条件参考)
        context_mask[n_sample:] = 0.0 

        self.nn_model.eval()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            x_i_repeat = x_i.repeat(2, 1, 1, 1)
            t_is_repeat = t_is.repeat(2, 1, 1, 1)
            
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i_repeat, c_i, t_is_repeat, context_mask)
            
            # 因为上面 Mask 的前半部分是 1，所以 eps1 是“有条件”
            eps1 = eps[:n_sample] 
            # 后半部分是 0，所以 eps2 是“无条件”
            eps2 = eps[n_sample:] 
            
            # 正向引导公式: Cond + w * (Cond - Uncond)
            eps = (1+guide_w)*eps1 - guide_w*eps2

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if return_all_steps:
                frames.append(x_i.clone())

        if return_all_steps:
            return torch.stack(frames, dim=0)
        else:
            return x_i