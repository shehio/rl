import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Tuple


class CNNMultiAction(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_layers_count: int,
        output_count: int,
        network_file: str,
        game_name: str,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels  # Number of color/feature channels
        self.hidden_layers_count = hidden_layers_count
        self.output_count = output_count
        self.game_name = game_name.replace("/", "_").replace("-", "_")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CNNMultiAction initialized on device: {self.device}")

        # CNN Architecture for 80x80 input
        # Input: (batch_size, input_channels, 80, 80)
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=8, stride=4, padding=0
        )  # 80->19
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)  # 19->8
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)  # 8->6

        # Calculate the size after convolutions
        conv_output_size = 64 * 6 * 6  # 2304 features

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_layers_count)
        self.fc2 = nn.Linear(hidden_layers_count, output_count)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Better initialization
        self._initialize_weights()

        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4, alpha=0.99, eps=1e-8)
        self.gradient_buffer = []
        self.network_file = network_file
        self.to(self.device)
        print(
            f"CNN Model: {input_channels}ch->32->64->64->{hidden_layers_count}->{output_count}"
        )

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_channels, 80, 80)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 2304)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def forward_pass(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # preprocess_pacman_frame_color_aware stacks features with axis=-1 and
        # ravels, so the flat vector is channels-LAST (80, 80, C). Reshaping it
        # straight to (1, C, 80, 80) reinterprets interleaved pixels as whole
        # channel planes and scrambles the conv input. Unpack in the layout it
        # was written in, then move channels to the front.
        input_reshaped = np.ascontiguousarray(
            input.reshape(1, 80, 80, self.input_channels).transpose(0, 3, 1, 2)
        )
        input_tensor = torch.from_numpy(input_reshaped).float().to(self.device)

        with torch.no_grad():
            output = self.forward(input_tensor)

            # Get hidden layer (after conv layers, before fc2)
            x = self.relu(self.conv1(input_tensor))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = x.view(x.size(0), -1)  # Flatten
            hidden_layer = self.relu(self.fc1(x))

        output_cpu = output.cpu().numpy().squeeze()  # Remove batch dimension
        hidden_cpu = hidden_layer.cpu().numpy().squeeze()  # Remove batch dimension

        return output_cpu, hidden_cpu

    def backward_pass(
        self, eph: np.ndarray, epdlogp: np.ndarray, epx: np.ndarray
    ) -> None:
        self.gradient_buffer.append((epx, eph, epdlogp))

    def train(self, learning_rate: float, decay_rate: float) -> None:
        if not self.gradient_buffer:
            return

        self.optimizer.param_groups[0]["lr"] = learning_rate
        self.optimizer.param_groups[0]["alpha"] = decay_rate
        self.optimizer.zero_grad()

        all_inputs = []
        all_hidden = []
        all_advantages = []

        for epx, eph, epdlogp in self.gradient_buffer:
            all_inputs.append(epx)
            all_hidden.append(eph)
            all_advantages.append(epdlogp)

        # Stack inputs and reshape for CNN
        inputs = np.vstack(all_inputs)  # Shape: (batch_size, 80 * 80 * input_channels)
        # Channels-last -> channels-first, same as forward_pass.
        inputs = np.ascontiguousarray(
            inputs.reshape(-1, 80, 80, self.input_channels).transpose(0, 3, 1, 2)
        )

        inputs = torch.from_numpy(inputs).float().to(self.device)
        hidden = torch.from_numpy(np.vstack(all_hidden)).float().to(self.device)
        advantages = torch.from_numpy(np.vstack(all_advantages)).float().to(self.device)

        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("🚨 CRITICAL: NaN or infinite inputs detected! Skipping training.")
            self.gradient_buffer = []
            return

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("🚨 CRITICAL: NaN or infinite advantages detected! Skipping training.")
            self.gradient_buffer = []
            return

        output = self.forward(inputs)
        output = torch.clamp(output, 1e-7, 1.0 - 1e-7)

        # Categorical cross-entropy loss for policy gradient
        loss = -torch.sum(torch.log(output) * advantages)

        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 CRITICAL: NaN or infinite loss detected! Skipping training.")
            self.gradient_buffer = []
            return

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.gradient_buffer = []

    def load_network(self, episode_number: int) -> None:
        if episode_number > 0:
            base_name = os.path.splitext(self.network_file)[0]
            file_name = f"{base_name}_{self.game_name}_cnn_ch{self.input_channels}_h{self.hidden_layers_count}_o{self.output_count}_{episode_number}"

            # Look in new models directory structure
            models_dir = f"../../models/pg/{self.game_name}"
            file_path = os.path.join(models_dir, file_name)

            if os.path.exists(file_path):
                state_dict = torch.load(file_path, map_location=self.device)
                self.load_state_dict(state_dict)
                print(f"Loaded CNN model: {file_path}")
            else:
                print(f"CNN model file not found: {file_path}")

    def save_network(self, episode_number: int) -> None:
        base_name = os.path.splitext(self.network_file)[0]
        file_name = f"{base_name}_{self.game_name}_cnn_ch{self.input_channels}_h{self.hidden_layers_count}_o{self.output_count}_{episode_number}"

        # Ensure the new models directory exists
        models_dir = f"../../models/pg/{self.game_name}"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)

        # Save in new models directory
        file_path = os.path.join(models_dir, file_name)
        torch.save(self.state_dict(), file_path)
