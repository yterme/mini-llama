import math
from torch import nn
import torch

IMAGE_SHAPE = (224, 224)
PATCH_SIZE = 16


class ViTTokenizer(nn.Module):
    def __init__(
        self, embed_dim, image_shape=IMAGE_SHAPE, patch_size=PATCH_SIZE, num_channels=3
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        # self.linear = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, embed_dim)
        self.W = nn.Parameter(torch.randn(self.num_channels * self.patch_size**2, embed_dim))
        # learnable position embedding
        num_tokens = math.ceil(image_shape[0] / self.patch_size) * math.ceil(
            image_shape[1] / self.patch_size
        )
        self.pos_embedding = nn.Parameter(torch.randn(num_tokens, embed_dim))

    # def pad_image(self, x):
    #     padding = (0, PATCH_SIZE - x.shape[2] % PATCH_SIZE, 0, PATCH_SIZE - x.shape[3] % PATCH_SIZE)
    #     if padding[1] == PATCH_SIZE:
    #         padding = (0, 0, 0, padding[3])
    #     if padding[3] == PATCH_SIZE:
    #         padding = (0, padding[1], 0, 0)
    #     x_padded = torch.nn.functional.pad(x, padding)
    #     return x_padded

    def __call__(self, x):
        # x = self.pad_image(x)
        # cut image x into non overlapping patches of size PATCH_SIZE
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # flatten the patches
        x = x.reshape(
            x.shape[0],
            x.shape[2],
            x.shape[3],
            x.shape[1],
            self.patch_size,
            self.patch_size,
        )
        x = x.flatten(start_dim=3, end_dim=5)
        x = x.flatten(start_dim=1, end_dim=2)
        x = x @ self.W
        # print(x.shape)
        # add position embedding
        x = x + self.pos_embedding
        return x


class GPTTokenizer(nn.Module):
    def __init__(self, vocab_size, embed_dim) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(vocab_size, embed_dim))
        # learnable position embedding
        # self.pos_embedding = nn.Parameter(torch.randn(num_tokens, embed_dim))

    def __call__(self, x):
        # x = self.pad_image(x)
        # cut image x into non overlapping patches of size PATCH_SIZE
        # x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # flatten the patches
        x = x.reshape(
            x.shape[0],
            x.shape[2],
            x.shape[3],
            x.shape[1],
            self.patch_size,
            self.patch_size,
        )
        x = x.flatten(start_dim=3, end_dim=5)
        x = x.flatten(start_dim=1, end_dim=2)
        x = x @ self.W
        # print(x.shape)
        # add position embedding
        x = x + self.pos_embedding
        return x
