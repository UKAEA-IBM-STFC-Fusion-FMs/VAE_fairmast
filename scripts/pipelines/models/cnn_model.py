import torch
import torch.nn as nn



# ======================================================================================================================
class Numerical0DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape):
        super().__init__()

        self.n_profiles = input_shape[0]

        self.bn = nn.BatchNorm1d(self.n_profiles)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        return self.bn(x)

# ======================================================================================================================
class Conv1DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D):
        super().__init__()

        self.n_profiles = input_shape[0]

        self.cnn = nn.Sequential(
            nn.BatchNorm1d(self.n_profiles),
            nn.Conv1d(self.n_profiles, D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(D),            
            nn.MaxPool1d(2, padding=1),
            nn.Conv1d(D, 2 * D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * D),
            nn.MaxPool1d(2, padding=1),
            nn.Conv1d(2 * D, 4 * D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(4 * D),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4 * D, self.n_profiles)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        x = self.cnn(x)
        x = self.global_avg_pool(x).squeeze(-1)  # [B, C]
        x = self.fc(x)
        return x

# ======================================================================================================================
class Conv2DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D):
        super().__init__()

        self.n_profiles = input_shape[0]
        
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(self.n_profiles),
            nn.Conv2d(self.n_profiles, D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(D),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(D, 2 * D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * D),
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(2 * D, 4 * D, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * D),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * D, self.n_profiles)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        x = self.cnn(x)
        x = self.global_avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        x = self.fc(x)
        return x


# ======================================================================================================================
class DecoderNumerical0DBranch(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, merged_input_dim, output_shape, D):
        super().__init__()

        self.n_timeseries = output_shape[0]

        self.fc = nn.Sequential(
            nn.BatchNorm1d(merged_input_dim),
            nn.Linear(merged_input_dim, 4 * D),
            nn.ReLU(),
            nn.BatchNorm1d(4 * D),
            nn.Linear(4 * D, 2 * D),
            nn.ReLU(),
            nn.BatchNorm1d(2 * D),
            nn.Dropout(0.2),
            nn.Linear(2 * D, self.n_timeseries),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        x = self.fc(x)
        return x

# ======================================================================================================================
class Conv1DDecoder(nn.Module):
    def __init__(self, merged_input_dim, output_shape, D):
        super().__init__()

        self.D = D
        self.n_profiles = output_shape[0]
        self.height_profiles = output_shape[1]

        self.fc = nn.Linear(merged_input_dim, 4 * D * self.height_profiles )   # match encoder bottleneck
        
        self.transposecnn = nn.Sequential(
            nn.ConvTranspose1d(4 * D, 2 * D, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(2 * D),
            nn.ConvTranspose1d(2 * D, D, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(D),
            nn.ConvTranspose1d(D, self.n_profiles, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_profiles),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(self.height_profiles)

    def forward(self, x):
        x = self.fc(x)                      # [B, 4D]
        x = x.view(-1, 4*self.D, self.height_profiles)     
        x = self.transposecnn(x)                   # [B, output_len, L]
        x = self.global_avg_pool(x)
        return x
    

# ======================================================================================================================
class Conv2DDecoder(nn.Module):
    def __init__(self, merged_input_dim, output_shape, D):
        super().__init__()

        self.D = D
        self.n_profiles = output_shape[0]
        self.height_profiles = output_shape[1]
        self.weight_profiles = output_shape[2]

        self.fc = nn.Linear(merged_input_dim, 4 * D * self.height_profiles * self.weight_profiles )   # match encoder bottleneck
        
        self.transposecnn = nn.Sequential(
            nn.ConvTranspose2d(4 * D, 2 * D, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * D),
            nn.ConvTranspose2d(2 * D, D, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(D),
            nn.ConvTranspose2d(D, self.n_profiles, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_profiles),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((self.height_profiles, self.weight_profiles))

    def forward(self, x):
        x = self.fc(x)                      # [B, 4D]
        x = x.view(-1, 4*self.D, self.height_profiles, self.weight_profiles)     
        x = self.transposecnn(x)                   # [B, output_len, L]
        x = self.global_avg_pool(x)
        return x


class MultiBranchCNNModel(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shapes, output_shapes, D=16):
        super().__init__()

        self.input_branches = nn.ModuleList()
        merged_input_dim = 0

        for shape in input_shapes:
            if len(shape) == 3:  # e.g., (2, 15, 17)
                branch = Conv2DBranch(shape, D)
                merged_input_dim += shape[0]
            elif len(shape) == 2:  # e.g., (1, 15)
                branch = Conv1DBranch(shape, D)
                merged_input_dim += shape[0]
            elif len(shape) == 1:  # e.g., (7,)
                branch = Numerical0DBranch(shape)
                merged_input_dim += shape[0]
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.input_branches.append(branch)
        
        # self.output_shape = output_shape[0][0]
        self.output_branches = nn.ModuleList()

        for shape in output_shapes:
            if len(shape) == 3:  # e.g., (2, 15, 17)
                # print("2D decoding branch for ", shape)
                branch = Conv2DDecoder(merged_input_dim, shape, D)
            elif len(shape) == 2:  # e.g., (1, 15)
                # print("1D decoding branch for ", shape)
                branch = Conv1DDecoder(merged_input_dim, shape, D)
            elif len(shape) == 1:  # e.g., (7,)
                # print("0D decoding branch for ", shape)
                branch = DecoderNumerical0DBranch(merged_input_dim, shape, D)
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.output_branches.append(branch)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *inputs):
        
        encoded_representation = []

        for branch, x in zip(self.input_branches, inputs):
            out = branch(x)
            encoded_representation.append(out)

        merged = torch.cat(encoded_representation, dim=1)

        decoded_representation = []

        for branch in self.output_branches :
            out = branch(merged)
            decoded_representation.append(out)
        
        return decoded_representation

    # ------------------------------------------------------------------------------------------------------------------


# ======================================================================================================================
# if __name__ == "__main__":
    
#     D = 16
#     B=32

#     input_shapes = [(6,), (2, 12), (3, 8), (3, 65, 65)]
#     input = ([torch.randn((B,) + shape) for shape in input_shapes])

#     output_shapes = [(7,), (1, 23), (1, 23, 45), (22, 76, 45)]
    
#     model = MultiBranchCNNModel(input_shapes, output_shapes, D)

#     print( "\nINPUT SHAPES: ", [ arr.shape for arr in input ] )
#     print( "\nOUTPUT SHAPES: ", [ arr.shape for arr in model(*input) ] )

#     from torchinfo import summary
#     summary(model, input_size=((32, 6,), (32, 2, 12), (32, 3, 8), (32, 3, 65, 65)))
