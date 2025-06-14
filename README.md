# Convolutional KAN for Image Classification: Overview and Improvement

- This repository provides a collection of Kolmogorov-Arnold Network (KAN) inspired layer implementations in PyTorch. It includes various KAN types, focusing on both Multi-Layer Perceptron (MLP) and Convolutional (1D, 2D, 3D) variants. 

## Features

This library implements several KAN-like layers, including:

*   **Spline-Based:**
    *   `KANLayer` / `KANConvNDLayer`: Based on B-spline activations (original KAN concept).
    *   `FastKANLayer` / `FastKANConvNDLayer`: Efficient KAN using Radial Basis Functions (RBFs).
*   **Polynomial-Based:**
    *   `LegendreKANLayer` / `LegendreKANConvNDLayer`: Uses Legendre polynomials.
    *   `ChebyKANLayer` / `ChebyKANConvNDLayer`: Uses Chebyshev polynomials.
    *   `TaylorKANLayer` / `TaylorKANConvNDLayer`: Uses Taylor series expansions.
    *   `BersnsteinKANLayer` / `BersnsteinKANConvNDLayer`: Uses Bernstein polynomials.
    *   `BesselKANLayer` / `BesselKANConvNDLayer`: Uses Bessel polynomials.
    *   `FibonacciKANLayer` / `FibonacciKANConvNDLayer`: Uses Fibonacci polynomials.
    *   `FourierKANLayer` / `FourierKANConvNDLayer`: Uses Fourier series.
    *   `GegenbauerKANLayer` / `GegenbauerKANConvNDLayer`: Uses Gegenbauer polynomials.
    *   `GRAMKANLayer` / `GRAMKANConvNDLayer`: Based on Gram polynomials.
    *   `HermiteKANLayer` / `HermiteKANConvNDLayer`: Uses Hermite polynomials.
    *   `JacobiKANLayer` / `JacobiKANConvNDLayer`: Uses Jacobi polynomials.
    *   `LaguerreKANLayer` / `LaguerreKANConvNDLayer`: Uses Laguerre polynomials.
    *   `LucasKANLayer` / `LucasKANConvNDLayer`: Uses Lucas polynomials.
*   **Wavelet-Based:**
    *   `WavKANLayer` / `WavKANConvNDLayer`: Uses various wavelet functions (Mexican Hat, Morlet, DoG, Meyer, Shannon). Includes different efficient convolutional implementations (`base`, `fast`, `fast_plus_one`).
*   **ReLU-Based:**
    *   `ReLUKANLayer` / `ReLUKANConvNDLayer`: Inspired by ReLU-based KAN approaches.

*   **Helper Factories:**
    *   `MLP_KAN_FACTORY` (in `kans.py`): Easily create MLP models using different KAN layer types.
    *   `CONV_KAN_FACTORY` (in `kan_conv.py`): Easily create KAN convolutional layers (primarily 2D focused in the factory).
*   **Example Models:**
    *   VGG-style (`kan_vgg.py`)
    *   AlexNet-style (`kan_alexnet.py`)
    *   EfficientNet-style (`kan_efficientnet.py`)
    *   EfficientNetV2-style (`kan_efficientnetv2.py`)
    *   MobileNetV1-style (`kan_mobilenet.py`)
    *   MobileNetV2-style (`kan_mobilenetv2.py`)
    *   MobileNetV3-style (`kan_mobilenetv3.py`)
    *   These models can be configured to use standard ConvNets or various KAN convolutional layers and different KAN or Linear classifiers.
*   **Utilities:**
    *   Regularization techniques (`regularization.py`).
    *   Dataloader (`dataloader.py`).
    *   Training script (`train.py`) with evaluation (`evaluations.py`, `generic_train.py`).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/GadGadGad/Convolutional-KAN-for-Image-Classification-Overview-and-Improvement
    cd Convolutional-KAN-for-Image-Classification-Overview-and-Improvement
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(You might need to create a `requirements.txt` file based on the imports)*:
    ```txt
    torch
    torchvision
    numpy
    scikit-learn
    tqdm
    matplotlib
    pandas
    huggingface_hub
    einops 
    ```

## Usage

### 1. Importing Layers Directly

You can import and use specific KAN layers like standard PyTorch modules.

```python
import torch
from layers import KANLayer, KANConv2DLayer, FastKANLayer, LegendreKANConv1DLayer

# Example: Standard KAN MLP Layer
mlp_layer = KANLayer(input_features=128, output_features=256, grid_size=5, spline_order=3)
input_mlp = torch.randn(64, 128) # Batch size 64, features 128
output_mlp = mlp_layer(input_mlp)
print("MLP Output Shape:", output_mlp.shape)

# Example: Standard KAN Conv2D Layer
conv_layer = KANConv2DLayer(input_dim=3, output_dim=16, kernel_size=3, spline_order=3, grid_size=5, padding=1)
input_conv = torch.randn(16, 3, 32, 32)

output_conv = conv_layer(input_conv)
print("Conv2D Output Shape:", output_conv.shape)
```

### 2. Using MLP Factory

The `kans.py` file provides a factory to easily create multi-layer KAN networks.

```python
import torch
from models.kans import MLP_KAN_FACTORY

# Example: Create a FastKAN MLP
layers_hidden = [784, 128, 64, 10] # Input, hidden dims, output
fast_kan_mlp = MLP_KAN_FACTORY['FastKAN'](
    layers_hidden=layers_hidden,
    grid_size=8,
    grid_range=[-2, 2]
)
input_tensor = torch.randn(32, 784) # Batch 32, features 784
output = fast_kan_mlp(input_tensor)
print("FastKAN MLP Output Shape:", output.shape)

# Example: Create a LegendreKAN MLP
legendre_kan_mlp = MLP_KAN_FACTORY['LegendreKAN'](
    layers_hidden=layers_hidden,
    degree=4
)
output_legendre = legendre_kan_mlp(input_tensor)
print("LegendreKAN MLP Output Shape:", output_legendre.shape)
```

### 3. Using Convolutional Layer Factory

The `kan_conv.py` file provides a factory (`CONV_KAN_FACTORY`) for creating various 2D KAN convolutional layers, handling padding calculation automatically.

```python
import torch
from layers.kan_conv import CONV_KAN_FACTORY

# Example: Create a WavKAN Conv2D layer
wav_conv = CONV_KAN_FACTORY['WavKAN'](
    in_planes=3,
    out_planes=32,
    kernel_size=3,
    wavelet_type='mexican_hat',
    wav_version='fast'
    # norm_layer=torch.nn.BatchNorm2d # Optional
)
input_tensor = torch.randn(16, 3, 64, 64)
output = wav_conv(input_tensor)
print("WavKAN Conv2D Output Shape:", output.shape)

# Example: Create a standard Conv2D layer using the same factory interface
std_conv = CONV_KAN_FACTORY['conv'](
    in_planes=3,
    out_planes=32,
    kernel_size=3
    # norm_layer=torch.nn.BatchNorm2d # Optional
    # base_activation=torch.nn.ReLU # Optional
)
output_std = std_conv(input_tensor)
print("Standard Conv2D Output Shape:", output_std.shape)
```

### 4. Using Pre-defined Models

This repository includes implementations of standard vision architectures adapted to use KAN layers.

```python
import torch
from models import VGGKAN, MobileNetV2KAN # etc.

model_vgg_kan = VGGKAN(
    input_channels=3,
    num_classes=10,
    arch='VGG16_small',
    conv_type='kanconv',     # Use KAN convolutions
    kan_conv='FastKAN',      # Specifically FastKAN convolutions
    classifier_type='Linear', # Use a standard Linear classifier head
    grid_size=8,             # Specific param for FastKAN conv layers
)

model_mobilenet_kanhead = MobileNetV2KAN(
    input_channels=3,
    num_classes=100,
    arch='default',
    conv_type='conv',           # Use standard convolutions
    classifier_type='KAN',      # Use a KAN classifier head
    kan_classifier='LegendreKAN', # Specifically LegendreKAN MLP layers
    classifier_degree=4,        # Specific param for LegendreKAN MLP head
)

input_tensor = torch.randn(4, 3, 224, 224) # Example input
output_vgg = model_vgg_kan(input_tensor)
output_mobile = model_mobilenet_kanhead(input_tensor)

print("VGG-KAN Output Shape:", output_vgg.shape)
print("MobileNetV2-KANHead Output Shape:", output_mobile.shape)
```

## Training

The `train.py` script provides a command-line interface to train the implemented models (VGG, AlexNet, EfficientNets, MobileNets, etc,.).

```bash
python train.py \
    --model VGGKAN \
    --arch default \
    --dataset CIFAR10 \
    --conv_type kanconv \
    --kan_conv KAN \
    --classifier_type KAN \
    --kan_classifier KAN \
    --batch_size 128 \
    --arch "VGG16" \
    --epoch 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grid_size 8 \
    --degree 4 \
    --norm_layer BatchNorm2d \
    --kan_norm_layer InstanceNorm2d \
    --results_dir ./training_results \
    --use_cuda
```

See `python train.py --help` for a full list of options to configure the model architecture, KAN parameters, training hyperparameters, and dataset choices.

## References

This repository builds upon the ideas ands code structures from the following repositories:
*   [TorchConv KAN](https://github.com/IvanDrokin/torch-conv-kan)
*   [OrthoPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs)
*   [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN)
*   [Wav-KAN](https://github.com/zavareh1/Wav-KAN)
*   [Efficient-KAN](https://github.com/Blealtan/efficient-kan)
*   [GRAMKAN](https://github.com/Khochawongwat/GRAMKAN)
*   [FourierKAN](https://github.com/GistNoesis/FourierKAN)
*   [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN)

We thank the authors of these repositories and papers for their valuable contributions to the field and for making their work publicly available.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
