
import torch
import torchvision
from torchvision import transforms

from tqdm import tqdm

from unet import UNet
from diffuser import NoiseDiffuser



def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, diffuser, totalTrainingTimesteps):
    """
    model: Object of Unet Model to train
    train_loader: Training batches of the total data
    val_loader: Validation batches of the total data
    optimizer: The backpropagation technique
    criterion: Loas Function
    device: CPU or GPU
    num_epochs: total number of training loops
    diffuser: NoiseDiffusion class object to perform Forward diffusion
    totalTrainingTimesteps: Total number of forward diffusion timesteps the model is to be trained on
    """

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Wrapping your loader with tqdm to display progress bar
        train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                  desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for batch_idx, data in train_progress_bar:
            data = data.to(device)
            optimizer.zero_grad()

            # Use a random time step for training
            batch_size = len(data)
            timesteps = torch.randint(0, totalTrainingTimesteps, (batch_size,), device=device).long()

            noisy_data, true_noise = diffuser.noise_diffusion(data, timesteps)
            predicted_noise = model(noisy_data, t=timesteps.float().view(-1, 1))

            loss = criterion(predicted_noise, true_noise)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({'Train Loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        # Wrapping your validation loader with tqdm to display progress bar
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader),
                                desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch_idx, data in val_progress_bar:
                data = data.to(device)

                # For simplicity, we can use the same random timestep for validation
                batch_size = len(data)
                timesteps = torch.randint(0, totalTrainingTimesteps, (batch_size,), device=device).long()

                noisy_data, true_noise = diffuser.noise_diffusion(data, timesteps)
                predicted_noise = model(noisy_data, t=timesteps.float().view(-1, 1))

                loss = criterion(predicted_noise, true_noise)
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses


total_timesteps = 300
startBeta, endBeta = 0.0001,0.2
inputChannels, outputChannels = 3, 3
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
full_trainset = torchvision.datasets.CelebA(root='./data', train=True, download=True, transform=transform)

# Splitting the trainset into training and validation datasets
train_size = int(0.8 * len(full_trainset))  # 80% for training
val_size = len(full_trainset) - train_size  # remaining 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(full_trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

DiffusionModel = UNet(inputChannels, outputChannels)
optimizer = torch.optim.Adam(DiffusionModel.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
diffuser = NoiseDiffuser(startBeta, endBeta, total_timesteps, device)


DiffusionModel = DiffusionModel.to(device)
train_losses, val_losses = train(model= DiffusionModel,
                                 train_loader= trainloader,
                                 val_loader= valloader,
                                 optimizer= optimizer,
                                 criterion= criterion,
                                 device= device,
                                 num_epochs= num_epochs,
                                 diffuser= diffuser,
                                 totalTrainingTimesteps=total_timesteps)

# Save the model
torch.save(DiffusionModel.state_dict(), 'ddpm.pth')

total_timesteps = 300
startBeta, endBeta = 0.0001,0.2
inputChannels, outputChannels = 3, 3
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DiffusionModel = UNet(inputChannels, outputChannels)
optimizer = torch.optim.Adam(DiffusionModel.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
diffuser = NoiseDiffuser(startBeta, endBeta, total_timesteps, device)


DiffusionModel = DiffusionModel.to(device)
train_losses, val_losses = train(model= DiffusionModel,
                                 train_loader= trainloader,
                                 val_loader= valloader,
                                 optimizer= optimizer,
                                 criterion= criterion,
                                 device= device,
                                 num_epochs= num_epochs,
                                 diffuser= diffuser,
                                 totalTrainingTimesteps=total_timesteps)

# Save the model
torch.save(DiffusionModel.state_dict(), 'ddpm.pth')

