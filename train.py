import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

class LunarHeightmapDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('I')
        
        image_np = np.array(image).astype(np.float32)
        image_np = (image_np / 32767.5) - 1.0
        
        image = Image.fromarray(image_np)
        
        if self.augment:
            image = self.apply_augmentation(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image

    def apply_augmentation(self, image):
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        rotation = np.random.choice([0, 90, 180, 270])
        if rotation != 0:
            image = image.rotate(rotation)
        
        return image

    def get_metadata(self):
        # Get dimensions of the first image
        first_image_path = os.path.join(self.data_dir, self.image_files[0])
        with Image.open(first_image_path) as img:
            width, height = img.size
        
        return {
            "num_original_images": len(self.image_files),
            "image_dimensions": f"{width}x{height}",
            "augmentation_enabled": self.augment
        }

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.model(input)
        output = output.view(-1, 1).squeeze(1)
        return self.sigmoid(output)


# Visualization function
def visualize_dataset(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    for i in range(num_images):
        idx = np.random.randint(len(dataset))
        img = dataset[idx].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    plt.tight_layout()
    plt.show()

def display_metadata(dataset, dataloader):
    metadata = dataset.get_metadata()
    print("\n=== Dataset Metadata ===")
    print(f"Number of original images: {metadata['num_original_images']}")
    print(f"Image dimensions: {metadata['image_dimensions']}")
    print(f"Augmentation enabled: {metadata['augmentation_enabled']}")
    
    if metadata['augmentation_enabled']:
        # Calculate the number of possible augmentations per image
        num_augmentations_per_image = 16  # 2 (horizontal flip) * 2 (vertical flip) * 4 (rotations)
        estimated_augmented_images = metadata['num_original_images'] * num_augmentations_per_image
        print(f"Estimated number of possible augmented images: {estimated_augmented_images}")
        print(f"Augmentation factor: {num_augmentations_per_image}x")
    
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    print("===========================\n")



# Training function
def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device, save_dir):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, label_real)

            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, label_fake)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            output = discriminator(fake_images)
            g_loss = criterion(output, label_real)

            g_loss.backward()
            g_optimizer.step()

            # Print progress
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

        # Save images every 10 epochs
        if (epoch + 1) % 5 == 0:
            save_image(fake_images.detach()[:25], f'{save_dir}/fake_images_epoch_{epoch+1}.png', nrow=5, normalize=True)

    print("Training complete!")

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 128
    batch_size = 32
    num_epochs = 300
    image_size = 512
    data_dir = r'HeightMaps\maps'
    save_dir = 'generated_images'

    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    dataset = LunarHeightmapDataset(data_dir, transform=transform, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Display metadata
    display_metadata(dataset, dataloader)

    # Visualize dataset
    visualize_dataset(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    print("Debugging information:")
    sample_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    sample_generated = generator(sample_noise)
    print(f"Generator output shape: {sample_generated.shape}")

    sample_real = next(iter(dataloader)).to(device)
    print(f"Real image shape: {sample_real.shape}")

    d_out_real = discriminator(sample_real)
    d_out_fake = discriminator(sample_generated)
    print(f"Discriminator output shape (real): {d_out_real.shape}")
    print(f"Discriminator output shape (fake): {d_out_fake.shape}")

    # Verify label shapes
    label_real = torch.ones(batch_size, device=device)
    label_fake = torch.zeros(batch_size, device=device)
    print(f"Label shapes - real: {label_real.shape}, fake: {label_fake.shape}")

    train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device, save_dir)

    torch.save(generator.state_dict(), f'{save_dir}/generator.pth')
    torch.save(discriminator.state_dict(), f'{save_dir}/discriminator.pth')