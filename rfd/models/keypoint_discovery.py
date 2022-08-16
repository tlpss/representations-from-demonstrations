import torch
from torch import nn
import pytorch_lightning as pl
from rfd.models.backbones.unet import UnetBackbone

class KeypointDetector(nn.Module):
    def __init__(self, n_keypoint_channels: int) -> None:
        super().__init__()

        self.backbone = UnetBackbone(n_channels_in=3,n_downsampling_layers=3,n_resnet_blocks=4,n_channels=32,kernel_size=3)
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

        self.model = nn.Sequential(self.backbone, head,nn.Sigmoid())

    
    def forward(self,x):
        return self.model(x)

class KeypointDecoder(nn.Module):
    def __init__(self, n_keypoints: int) -> None:
        super().__init__()


class KeypointDiscovery(pl.LightningModule):
    def __init__(self, n_keypoints: int) -> None:
        super().__init__()

        self.keypoint_detector = KeypointDetector(n_keypoints)

        self.spatial_softmax = None
        self.decoder = None
    def foward(self, x):
        return self.keypoint_detector(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=3e-4)
    
    def training_step(self, batch, batch_idx):
        sources, targets = batch
        source_keypoints = self.keypoint_detector(sources)
        target_keypoints = self.keypoint_detector(targets)

        # extract spatial locations of the keypoints using a spatial softmax (differentiable)
        # bring back to gaussians (differentiable!)

        # concat (src_kps, target_kps, source_img) 
        # feed into decoder

        # compute loss.
    def spatial_softmax(x: torch.Tensor) -> torch.Tensor:
        # N x C x H x W -> N x C x 2 (x,y)
        

    def gaussian_heatmap(
        self,
        image_size: Tuple[int, int],
        center: torch.Tensor,
        sigma: torch.Tensor,
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

        # cast keypoints (center) to ints to make grid align with pixel raster.
        #  Otherwise, the AP metric for  d = 1 will not result in 1
        #  if the gt_heatmaps are used as input.
        u_axis = torch.linspace(0, image_size[1] - 1, image_size[1], device=self.device) - int(center[0])
        v_axis = torch.linspace(0, image_size[0] - 1, image_size[0], device=self.device) - int(center[1])
        # create grid values in 2D with x and y coordinate centered aroud the keypoint
        xx, yy = torch.meshgrid(v_axis, u_axis)
        xx = xx.unsqueeze(0) - centers.unsqueeze(-1).unsqueeze(-1).
        ## create gaussian around the centered 2D grids $ exp ( -0.5 (x**2 + y**2) / sigma**2)$
        heatmap = torch.exp(-0.5 * (torch.square(xx- center[0]) + torch.square(yy)) / torch.square(sigma))
        return heatmap
    
    def visualize(self):
        # get keypoints
        # overlay keypoints on original images
        # get reconstruction

        #original - kp1 - kp2 - reconstruction - target GRID
        pass


if __name__ == "__main__":
    k = KeypointDetector(3)
    x = torch.randn(4,3,128,128)
    y = k(x)
    print(y.shape)
