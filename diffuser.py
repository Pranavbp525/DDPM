
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseDiffuser:
    def __init__(self, start_beta, end_beta, total_steps, device='cpu'):
        assert start_beta < end_beta < 1.0

        self.device = device
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.total_steps = total_steps

        self.betas = self.linear_scheduler()
        self.alphas = 1 - self.betas
        self.alphas = self.alphas.to(device)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar = self.alpha_bar.to(device)

    def linear_scheduler(self):
        """Returns a linear schedule from start to end over the specified total number of steps."""

        return torch.linspace(self.start_beta, self.end_beta, self.total_steps).to(self.device)

    def noise_diffusion(self, image, t):
        """
        Diffuse noise into an image based on timestep t using the pre-computed cumulative product of alphas.
        """
        image = image.to(self.device)
        noise = torch.randn_like(image).to(self.device)
        alpha_bar_t = self.alpha_bar[t]  # t is now a tensor

        result = image * alpha_bar_t[:, None, None, None] + noise * (1 - alpha_bar_t[:, None, None, None])
        return result, noise
