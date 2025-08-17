import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

class DDPMWrapper(torch.nn.Module):
    def __init__(self, unet, num_train_timesteps=1000):
        super().__init__()
        self.unet = unet
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    @torch.no_grad()
    def add_noise(self, x0, noise, t):
        return self.scheduler.add_noise(x0, noise, t)

    def loss_step(self, x0, labels):
        """
        x0: (B,3,H,W) in [-1,1]
        labels: (B,24) multi-hot
        """
        B = x0.size(0)
        device = x0.device
        T = self.scheduler.config.num_train_timesteps
        t = torch.randint(0, T, (B,), device=device).long()
        noise = torch.randn_like(x0)
        x_t = self.scheduler.add_noise(x0, noise, t)
        eps_hat = self.unet(x_t, t, labels)
        loss = F.mse_loss(eps_hat, noise)
        return loss

    @torch.no_grad()
    def sample(self, labels, num_steps=None, batch_size=None, shape=(3,64,64), device=None):
        """
        labels: (B,24)
        Returns: x_0 in [-1,1], shape (B,3,H,W)
        """
        if device is None:
            device = labels.device
        C, H, W = shape
        B = labels.size(0) if batch_size is None else batch_size

        # Configure scheduler
        T = num_steps or self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(T, device=device)

        # Start from Gaussian noise
        x = torch.randn(B, C, H, W, device=device)

        for t in self.scheduler.timesteps:
            eps_hat = self.unet(x, t.expand(B), labels)
            x = self.scheduler.step(eps_hat, t, x).prev_sample
        return x
    
    @torch.no_grad()
    def p_sample(self, x_t, t, labels):
        eps = self.unet(x_t, t, labels)
        step = self.scheduler.step(eps, t, x_t)
        return step.prev_sample
