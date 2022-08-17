from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import seaborn as sns 

from rfd.models.backbones.unet import UnetBackbone
from rfd.utils import get_logging_path


class KeypointDetector(nn.Module):
    def __init__(self, n_keypoint_channels: int) -> None:
        super().__init__()

        self.backbone = UnetBackbone(
            n_channels_in=3, n_downsampling_layers=3, n_resnet_blocks=4, n_channels=32, kernel_size=3
        )
        self.n_channels_out = n_keypoint_channels

        head = nn.Conv2d(
            in_channels=self.backbone.get_n_channels_out(),
            out_channels=self.n_channels_out,
            kernel_size=(3, 3),
            padding="same",
        )

        # expect output of backbone to be normalized!
        # so by filling bias to -4, the sigmoid should be on avg sigmoid(-4) =  0.02
        # which is consistent with the desired heatmaps that are zero almost everywhere.
        # setting too low would result in loss of gradients..
        head.bias.data.fill_(-4)

        self.model = nn.Sequential(self.backbone, head, nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class KeypointDecoder(nn.Module):
    def __init__(self, n_keypoints: int, n_channels=32) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3 + 2 * n_keypoints, n_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(n_channels, 3, kernel_size=3, padding="same"),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.network(x)


class KeypointDiscovery(pl.LightningModule):
    def __init__(self, n_keypoints: int) -> None:
        super().__init__()

        self.keypoint_detector = KeypointDetector(n_keypoints)

        self.decoder = KeypointDecoder(n_keypoints)

    def forward(self, x):
        heatmaps = self.keypoint_detector(x)
        keypoints = self.extract_keypoints_from_heatmap(heatmaps,device=self.device)
        return keypoints

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)

    def shared_step(self, batch, batch_idx, is_validation_step=False):

        sources, targets = batch
        # get keypoints using the encoder

        source_heatmaps = self.keypoint_detector(sources)
        target_heatmaps = self.keypoint_detector(targets)

        source_keypoints = self.extract_keypoints_from_heatmap(source_heatmaps,device=self.device)
        target_keypoints = self.extract_keypoints_from_heatmap(target_heatmaps,device = self.device)

        # bring back to gaussians (differentiable!)
        img_shape = sources.shape[2:]
        source_gaussian_heatmaps = self.gaussian_heatmap(img_shape, source_keypoints,device=self.device)
        target_gaussian_heatmaps = self.gaussian_heatmap(img_shape, target_keypoints,device = self.device)
        # concat (src_kps, target_kps, source_img)

        decoder_input = torch.cat([sources, source_gaussian_heatmaps, target_gaussian_heatmaps], dim=1)
        # feed into decoder

        reconstructed_targets = self.decoder(decoder_input)

        loss = nn.functional.mse_loss(targets, reconstructed_targets)

        if batch_idx == 0:
            # visualize
            num_samples = min(8, sources.shape[0])
            self.visualize(
                sources[:num_samples].detach().cpu(),
                source_heatmaps[:num_samples].detach().cpu(),
                target_heatmaps[:num_samples].detach().cpu(),
                reconstructed_targets[:num_samples].detach().cpu(),
                targets[:num_samples].detach().cpu(),
                is_validation_step,
            )

        return loss

        # compute loss.

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, is_validation_step=False)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, is_validation_step=True)
        self.log("validation/loss", loss)
        return loss

    @staticmethod
    def extract_keypoints_from_heatmap(x: torch.Tensor, device = 'cpu') -> torch.Tensor:
        # N x C x H x W -> N x C x 2 (x,y)
        # separable implementation according to formula in https://rll.berkeley.edu/dsae/dsae.pdf

        # spatial soft(arg)max
        
        alpha = 0.02
        x = torch.exp(x / alpha)
        x = x / torch.sum(x, [-1, -2], keepdim=True)

        # get the center of mass / expected value (everything is connected http://math.oxford.emory.edu/site/math117/expectedValueAndCenterOfMass/)
        conditial_v_distribution = torch.sum(x, dim=-1)  # N x C x H
        conditional_u_distribution = torch.sum(x, dim=-2)  # N x C x W

        i = torch.linspace(0, x.shape[-1], x.shape[-1], device=device).unsqueeze(0).unsqueeze(0)  # 1 x 1 x W
        j = torch.linspace(0, x.shape[-2], x.shape[-2], device=device).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H

        expected_u = torch.sum(conditional_u_distribution * i, dim=-1)  # N x C x  W * 1 x 1 x W -> N x C
        expected_v = torch.sum(conditial_v_distribution * j, dim=-1)  # N x C x H * 1 x 1 x H -> N x C
        coords = torch.cat([expected_u.unsqueeze(-1), expected_v.unsqueeze(-1)], dim=-1)  # N x C x 2
        return coords

    @staticmethod
    def gaussian_heatmap(
        image_size: Tuple[int, int],
        centers: torch.Tensor,
        sigma: float = 4,
        device='cpu'
    ) -> torch.Tensor:
        """
        Creates a Gaussian blob heatmap for a single keypoint.
        The coordinate system is a left-top corner origin with u going right and v going down.
        Args:
            image_size (Tuple(int,int)): image_size (height, width) ! note convention to match store-order and tensor dimensions
            center (Tuple(int,int)): center coordinate (cX,cY) (U,V) ! note: U,V order
            sigma (torch.Tensor): standard deviation of the gaussian blob
        Returns:
            Torch.Tensor: A tensor with zero background, specified size and a Gaussian heatmap around the center.
        """

        u_axis = torch.linspace(0, image_size[1] - 1, image_size[1], device=device)  # W,
        v_axis = torch.linspace(0, image_size[0] - 1, image_size[0], device=device)  # W,
        # create grid values in 2D with x and y coordinate centered aroud the keypoint
        v_grid, u_grid = torch.meshgrid(v_axis, u_axis, indexing="ij")  # v-axis -> dim 0, u-axis -> dim 1
        u_grid = u_grid.unsqueeze(0).unsqueeze(0) - centers[..., 0].unsqueeze(-1).unsqueeze(
            -1
        )  # 1,1,H,W - N x C x 1 x 1
        v_grid = v_grid.unsqueeze(0).unsqueeze(0) - centers[..., 1].unsqueeze(-1).unsqueeze(
            -1
        )  # 1,1,H,W - N x C x 1 x 1

        ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
        heatmap = torch.exp(
            -0.5
            * (torch.square(u_grid) + torch.square(v_grid))
            / torch.square(torch.tensor([sigma], device=device))
        )
        return heatmap

    def visualize(self, sources, source_heatmaps, target_heatmaps, target_reconstructions, targets, is_validation_step: bool):
        # get keypoints

        num_samples = source_heatmaps.shape[0]
        source_heatmaps.shape[1]

        source_keypoints = self.extract_keypoints_from_heatmap(source_heatmaps)
        target_keypoints = self.extract_keypoints_from_heatmap(target_heatmaps)

        keypoint_overlayed_source_images = self.overlay_images_with_keypoints(sources,source_keypoints)
        keypoint_overlayed_target_images = self.overlay_images_with_keypoints(targets, target_keypoints)


        images = torch.cat([sources, keypoint_overlayed_source_images, target_reconstructions, keypoint_overlayed_target_images, targets])
        grid = torchvision.utils.make_grid(images, n_row=num_samples)


        key = "validation" if is_validation_step else "training"
        self.logger.log_image(key=key, images=[grid], caption=["todo"])

        # overlay keypoints on original images
        # get reconstruction

        # original - kp1 - kp2 - reconstruction - target GRID


    @staticmethod    
    def overlay_images_with_keypoints(images: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:

        # heatmaps N x C x H x W 
        # keypoints N x C x 2 

        n_keypoints = keypoints.shape[1]
        image_size = images.shape[2:]
        alpha = 0.7

        colors = torch.Tensor(sns.color_palette("husl",n_keypoints)) # C x 3
        colors = colors.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # 1 x C x 3 x 1 x 1 

        heatmaps = KeypointDiscovery.gaussian_heatmap(image_size,keypoints,sigma= 2) # N x C x H x W 
        heatmaps = heatmaps.unsqueeze(2) # N x C x 1 x H x W

        colorized_heatmaps = colors * heatmaps
        combined_heatmap = torch.max(colorized_heatmaps,dim=1)[0] # N x 3 x H x W 

        overlayed_images = images * alpha  + combined_heatmap 
        overlayed_images = torch.clip(overlayed_images,0.0,1.0)

        return overlayed_images



   
    



if __name__ == "__main__":
    # k = KeypointDetector(3)
    # x = torch.randn(4,3,128,128)
    # y = k(x)
    # print(y.shape)

    # k = KeypointDiscovery(4)

    # centers = torch.tensor([[[20,30],[90,10]],[[40,50],[100,30]]])
    # print(centers.shape)

    # heatmaps = k.gaussian_heatmap((100,120), centers, torch.tensor([1]))
    # print(centers[0][1])
    # plt.imsave("test.jpg",heatmaps[0][1])

    # print(heatmaps.shape)
    # kp = k.extract_keypoints_from_heatmap(heatmaps)
    # print(kp)

    # loss = k.training_step([torch.randn(5,3,256,128),torch.randn(5,3,256,128)],0)
    # print(loss)

    import pytorch_lightning as pl
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    from rfd.dataset.dataloader import RandomSplitDataModule
    from rfd.dataset.demonstration_dataset import PredictFutureFrameDataset
    from rfd.utils import get_data_path

    dataset = PredictFutureFrameDataset(get_data_path() / "push_demonstrations", 2)
    datamodule = RandomSplitDataModule(dataset, 32, 0.1, 4)
    model = KeypointDiscovery(5)

    wandb.init(project="learning-from-demonstrations", dir=get_logging_path(), tags = ["keypoint discovery"],mode="online")
    logger = WandbLogger(save_dir=get_logging_path())
    trainer = pl.Trainer(max_epochs=30, gpus=1, overfit_batches = 1, logger=logger)

    trainer.fit(model, datamodule)
