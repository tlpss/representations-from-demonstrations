from configparser import Interpolation
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb 
import torchvision

from rfd.utils import get_logging_path
class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 image_size: int,
                 n_channels : int,
                 latent_dim : int):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - n_channels : Number of channels for all other layers.
            - latent_dim : Dimensionality of latent representation z
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, n_channels, kernel_size=3, padding=1), #/2
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1), #/4
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1), #/8
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear((image_size // 8)**2*n_channels, latent_dim)

        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim:int, n_channels_hidden: int, n_channels_out: int, image_size:int) -> None:
        super().__init__()
        self.image_size = image_size
        self.linear = nn.Linear(latent_dim, (image_size//8)**2*n_channels_hidden)
        self.net = nn.Sequential(
            nn.Conv2d(n_channels_hidden, n_channels_hidden, kernel_size=3, padding=1), #/8 -> /4
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(n_channels_hidden, n_channels_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels_hidden, n_channels_hidden, kernel_size=3, padding=1), #/4 -> /2
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(n_channels_hidden, n_channels_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels_hidden, n_channels_hidden, kernel_size=3, padding=1), #/2 -> /1
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            nn.Conv2d(n_channels_hidden, n_channels_out, kernel_size=3, padding=1),
            nn.ReLU(),

        )

    def forward(self,x):
        x = self.linear(x)
        x = x.reshape(x.shape[0],-1, self.image_size//8, self.image_size//8)
        return self.net(x)



class AutoEncoder(pl.LightningModule):
    """very simple auto encoder to play around
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch, batch_idx) -> torch.Tensor:
        # pass trough encoder to obtain representation
        source,target = batch
        z = self.encoder(source)
        # then move through decoder
        reconstructed = self.decoder(z)
        # compute the loss
        loss = nn.functional.mse_loss(reconstructed,target)
        return loss

    def create_reconstruction_image_grid(self,source, target):
        with torch.no_grad():
            num_images = min(8, source.shape[0])
            reconstructed = self.decoder(self.encoder(source))
            images = torch.cat([source[:num_images,...],reconstructed[:num_images,...],target[:num_images,...]])
            grid = torchvision.utils.make_grid(images, n_row = num_images)

            return grid


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss)

        return loss


    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            source,target = batch[0], batch[1]
            grid = self.create_reconstruction_image_grid(source,target)

            self.logger.log_image(key=f"validation", images=[grid], caption=["top: source, middle: reconstructions, bottom: target"])
        loss = self.shared_step(batch, batch_idx)
        
        self.log("validation/loss",loss)
        return loss
    


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 3e-4)

    
if __name__ == "__main__":
    from rfd.dataset.demonstration_dataset import PredictFutureFrameDataset
    from rfd.dataset.dataloader import RandomSplitDataModule
    from rfd.utils import get_data_path
    enc = Encoder(3,128,32,8)
    dec = Decoder(8,32,3,128)

    autoencoder = AutoEncoder(enc,dec)

    dataset = PredictFutureFrameDataset(get_data_path() / "push_demonstrations",0)
    datamodule = RandomSplitDataModule(dataset, 32, 0.1,4)


    wandb.init(project="learning-from-demonstrations", dir=get_logging_path(), tags = ["autoencoder"])
    logger = WandbLogger(save_dir=get_logging_path())
    trainer = pl.Trainer(max_epochs=30, gpus=1, logger = logger)

    trainer.fit(autoencoder, datamodule)
