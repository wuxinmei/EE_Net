import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from math import ceil
import yacs.config
from ptflops import get_model_complexity_info
from DNN_printer import DNN_printer
from torchstat import stat

base_model = [
    # index, expand_ratio, channels, repeats, stride, kernel_size, padding
    # [0, 1, 96, 1, 3, 1, 1],
    [1, 6, 256, 1, 1, 5, 2],
    [2, 6, 384, 1, 1, 3, 1],
    [3, 6, 64, 1, 1, 1, 0],
]

phi_values = {
    # tuple of: (phi_value,width_coefficient, depth_coefficient, resolution, drop_rate)
    "b0": (0, 1.0, 1.0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 1.0, 1.1, 231, 0.25),
    "b2": (1, 1.1, 1.2, 240, 0.3),
    "b3": (1.5, 1.15, 1.3, 248, 0.35),
    "b4": (2, 1.2, 1.4, 257, 0.4),
    "b5": (2.5, 1.3, 1.6, 267, 0.4),
    "b6": (3, 1.4, 1.8, 276, 0.5),
    # "b6": (3, 1.4, 1.8, 340, 0.5),
    # "b7": (3.5, 1.6, 2.2, 380, 0.5),
}
# phi_values = {
#     # tuple of: (phi_value,width_coefficient, depth_coefficient, resolution, drop_rate)
#     "b0": (0, 1.0, 1.0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
#     "b1": (0.5, 1.0, 1.1, 240, 0.25),
#     "b2": (1, 1.1, 1.2, 260, 0.3),
#     "b3": (2, 1.2, 1.4, 300, 0.35),
#     "b4": (3, 1.4, 1.8, 380, 0.4),
#     "b5": (4, 1.6, 2.2, 456, 0.4),
#     "b6": (5, 1.8, 2.6, 528, 0.5),
#     "b7": (6, 2.0, 3.1, 600, 0.5),
#     "b8": (7, 2.2, 3.6, 672, 0.5),
# }


class CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1, image_size=224
    ):     ###  group=1 is a normal conv, groups=in_channnels it is a depthwise conv
        super(CNNBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cnn(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)    # C x H x W -> C x 1 x 1 squeeze block
        self.se = nn.Sequential(
            nn.Linear(in_channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4,  # squeeze excitation
            survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)
        self.count = 0

        # if self.expand:
        self.expand_conv = nn.Sequential(
            CNNBlock(
            in_channels, hidden_dim, kernel_size=1, stride=1, padding=0
            ),
        )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),

            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=50.001, momentum=0.010000000000000009),
            )


    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob

        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class EENet(nn.Module):
    def __init__(self, version):
        super(EENet, self).__init__()
        # width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        phi, width_factor, depth_factor, res, drop_rate = phi_values[version]
        # last_channels = 4*ceil(int(256*width_factor) / 4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            # nn.Linear(last_channels, num_classes),
        )

    # def calculate_factors(self, version, alpha=1.2, beta=1.1):
    #     phi, alpha, beta, res, drop_rate = phi_values[version]
    #     depth_factor = alpha ** phi
    #     width_factor = beta ** phi
    #     return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor):
        channels = 4*ceil(int(96*width_factor) / 4)
        features = [CNNBlock(3, channels, 11, stride=4, padding=0)]
        in_channels = channels

        for i, expand_ratio, channels, repeats, stride, kernel_size, padding in base_model:
            out_channels = 4*ceil(int(channels*width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride,   ## if layer == 0 else 1,
                        kernel_size=kernel_size,
                        # padding=kernel_size//2,   # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                        padding=padding,
                    )
                )

                if layer == 0 and i != 3:
                    features.append(nn.MaxPool2d(kernel_size=3, stride=2),)
                    # features.append(nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),)

                in_channels = out_channels

        return nn.Sequential(*features)


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return x

class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights) ### AlexNet
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

class FaceImageModel(nn.Module):

    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            # nn.Linear(12*12*64, 128),    # b0
            # nn.Linear(13*13*64, 128),    # b2/b1
            nn.Linear(14*14*64, 128),    # b4/b3
            # nn.Linear(15*15*64, 128),    # b5
            # nn.Linear(16*16*64, 128),    # b6
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize = 25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            # nn.Linear(12*12*64, 128),    #b0
            # nn.Linear(13*13*64, 128),    # b2/b1
            nn.Linear(14*14*64, 128),    # b4/b3
            # nn.Linear(15*15*64, 128),    # b5
            # nn.Linear(16*16*64, 128),    # b6
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            # nn.ReLU(inplace=True),
            )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sig(self.fc(x))
        return x

class Model(nn.Module):

    def __init__(self, config: yacs.config.CfgNode):
    # def __init__(self):
        super(Model, self).__init__()
        # self.eyeModel = ItrackerImageModel()
        self.version = "b4"
        self.eyeModel = EENet(self.version)
        phi, width_factor, depth_factor, res, drop_rate = phi_values[self.version]
        last_channels = 4*ceil(int(64*width_factor) / 4)
        self.faceModel = FaceImageModel()
        self.attention = AttentionNet()
        # self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2*last_channels, 128),
            nn.ReLU(inplace=True),
            )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+64+2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )


    def forward(self, faces, eyesLeft, eyesRight, poses):
        # Eye nets
        xEyeR = self.eyeModel(eyesRight)
        xEyeL = self.eyeModel(eyesLeft)
        Lweight = self.attention(eyesLeft)
        Rweight = self.attention(eyesRight)
        xEyeL = Lweight * xEyeL
        xEyeR = Rweight * xEyeR

        # # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)


        # Face net
        xFace = self.faceModel(faces)
        # xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, poses), 1)
        x = self.fc(x)

        return x

# def test():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # print(device)
#     version = "b2"
#     _, _, _, res, drop_rate = phi_values[version]

#     x1 = torch.randn(1, 3, 276, 276).to(device)
#     x2 = torch.randn(1, 3, 300, 300).to(device)
#     x3 = torch.randn(1, 3, 224, 224).to(device)
#     x4 = torch.randn(1, 1, 25, 25).to(device)
#     x5 = torch.randn(1, 2).to(device)

#     model1 = Model().to(device)
#     print(model1(x2, x2, x2, x5))
#     # model = ItrackerImageModel()
#     # print(model.fc[0].weight) # (num_examples, num_classes)
#     model2 = EENet(version=version).to(device)
#     print(model2(x1).shape)

# test()
# def test():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # print(device)
#     version = "b6"
#     phi, width_factor, depth_factor, res, drop_rate = phi_values[version]
#
#     x1 = torch.randn(1, 3, 276, 276).to(device)
#     x2 = torch.randn(1, 3, 300, 300).to(device)
#     x3 = torch.randn(1, 3, 224, 224).to(device)
#     x4 = torch.randn(1, 1, 25, 25).to(device)
#     x5 = torch.randn(1, 2).to(device)
#
#     model1 = Model().to(device)
#     # print(model1(x3, x1, x1, x5))
#     # model = ItrackerImageModel().to(device)
#     # print(model(x2).shape)
#     # print(model.fc[0].weight) # (num_examples, num_classes)
#     model2 = EENet(version=version)
#     # # x = model2(x1)
#     # print(model2)
#     # flops, params = get_model_complexity_info(model2, (3, res, res),
#     #                                           as_strings=True,
#     #                                           print_per_layer_stat=True,
#     #                                           verbose=True,
#     #                                           # input_constructor=True
#     #                                           )
#     # print("FLOPs: " + flops)
#     # print('{:<30} {:<8}'.format("number of paramaters: ", params))
#     # # DNN_printer(model1, [(3, 276, 276), (3, 276, 276), (3, 276, 276), (2, 1)], 1)
#     # model1 = ItrackerImageModel()
#
#     # stat(model, (3, 224, 224))
#     # print(model2)
#     stat(model2, (3, res, res))
# test()
# #
