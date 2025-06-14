# KAN-Based Vision Models

## Key Features

*   **KAN Integration:** Standard vision architectures modified to use KAN layers.
*   **Multiple KAN Variants:** Supports various KAN layer implementations for both convolutions (`layers/kan_conv.py`) and MLPs (`kans.py`), such as:
    *   Standard B-Spline KANs
    *   FastKAN
    *   Polynomial KANs (Legendre, Chebyshev, Bernstein, Jacobi, etc.)
    *   FourierKAN, WavKAN, ReLUKAN, etc.
*   **Configurable:** Allows customization of KAN parameters, network width, dropout, normalization layers, and more.

## Models Included

This module includes KAN-adapted versions of the following architectures:

*   **AlexNet:** (`kan_alexnet.py`)
*   **VGG:** (`kan_vgg.py`)
*   **MobileNetV1:** (`kan_mobilenet.py`)
*   **MobileNetV2:** (`kan_mobilenetv2.py`)
*   **MobileNetV3:** (`kan_mobilenetv3.py`)
*   **EfficientNet:** (`kan_efficientnet.py`)
*   **EfficientNetV2:** (`kan_efficientnetv2.py`)

Each model file typically provides a factory function (e.g., `mobilenet_v3_kan`) for easy instantiation. Please refer to the specific Python files and function signatures for detailed configuration options.

