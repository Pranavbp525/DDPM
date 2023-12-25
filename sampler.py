
import torch
import matplotlib.pyplot as plt
from diffuser import NoiseDiffuser
from unet import UNet


def display_images(images, n, images_per_row, labels=None):
    num_rows = (n + images_per_row - 1) // images_per_row  # Calculate the number of rows needed
    plt.figure(figsize=(2 * images_per_row, 2 * num_rows))  # Adjust the size as needed

    for i in range(n):
        plt.subplot(num_rows, images_per_row, i + 1)

        # Rearrange the tensor from (C, H, W) to (H, W, C)
        img = images[i].cpu().permute(1, 2, 0).squeeze()

        # If the image is grayscale (1 channel), then we use 'gray' colormap, otherwise use default
        cmap = 'gray' if img.shape[2] == 1 else None

        plt.imshow(img.numpy(), cmap=cmap)
        plt.axis('off')  # Turn off the axis
        if labels is not None:
            plt.title(labels[i])

    plt.tight_layout()
    plt.show()


def generate_samples(x_t, model, num_samples, total_timesteps, diffuser, device):
    with torch.no_grad():  # Reduces memory usage during inference
        one_by_sqrt_alpha = 1 / torch.sqrt(diffuser.alphas)
        beta_by_sqrt_one_minus_alpha_cumprod = diffuser.betas / torch.sqrt(1 - diffuser.alpha_bar)

        for timestep in range(total_timesteps - 1, -1, -1):
            z = torch.randn_like(x_t)
            ts = torch.ones(num_samples, dtype=torch.float32, device=device) * timestep
            epsilon_t = model(x_t, ts.view(-1, 1))
            x_t_minus_1 = (one_by_sqrt_alpha[timestep] * (
                        x_t - (beta_by_sqrt_one_minus_alpha_cumprod[timestep] * epsilon_t))) + (
                                      torch.sqrt(diffuser.betas[timestep]) * z)
            x_t = x_t_minus_1

        return x_t.detach()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffuser = NoiseDiffuser(start_beta=0.0001, end_beta=0.02, total_steps=300,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
total_timesteps = 300
# Using the function:
model_path = 'best_model.pth'
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

SEED = [96, 786, 7150]  # You can set any integer value for the seed

for S in SEED:
    print("The Outputs for Random Seed {%d}" % S)
    # Set seed for both CPU and CUDA devices
    torch.manual_seed(S)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(S)
        torch.cuda.manual_seed_all(S)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_samples_to_generate = 10
    # Initialize with random noise
    xt = torch.randn((num_samples_to_generate, 3, 128, 128), device=device)

    samples = generate_samples(xt, model, num_samples_to_generate, total_timesteps, diffuser, device)

    # Display the generated samples
    display_images(samples, num_samples_to_generate, images_per_row=5)
