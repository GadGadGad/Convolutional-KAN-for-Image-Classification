# KAN-Based Layers

This folders provides a collection of PyTorch modules implementing various Kolmogorov-Arnold Network (KAN) layers, including both standard linear layers and convolutional layers. The implementations are based on several recent papers and open-source projects, offering different mathematical bases for the learnable activation functions on the edges of the network.



**Note:** This project consolidates and adapts code from several sources (see Acknowledgements).

## Features

Provides implementations for various KAN types:

*   **Spline-Based:**
    *   `KANLayer` / `KANConvNDLayer`: Based on B-splines (Original KAN concept).
    *   `FastKANLayer` / `FastKANConvNDLayer`: Uses Radial Basis Functions for potentially faster computation.
*   **Polynomial-Based:**
    *   `TaylorKANLayer` / `TaylorKANConvNDLayer`: Uses Taylor series expansions.
    *   `ChebyKANLayer` / `ChebyKANConvNDLayer`: Uses Chebyshev polynomials.
    *   `LegendreKANLayer` / `LegendreKANConvNDLayer`: Uses Legendre polynomials.
    *   `LaguerreKANLayer` / `LaguerreKANConvNDLayer`: Uses Laguerre polynomials.
    *   `HermiteKANLayer` / `HermiteKANConvNDLayer`: Uses Hermite polynomials.
    *   `JacobiKANLayer` / `JacobiKANConvNDLayer`: Uses Jacobi polynomials.
    *   `GegenbauerKANLayer` / `GegenbauerKANConvNDLayer`: Uses Gegenbauer polynomials.
    *   `BersnsteinKANLayer` / `BersnsteinKANConvNDLayer`: Uses Bernstein polynomials.
    *   `BesselKANLayer` / `BesselKANConvNDLayer`: Uses Bessel polynomials.
    *   `FibonacciKANLayer` / `FibonacciKANConvNDLayer`: Uses Fibonacci polynomials.
    *   `LucasKANLayer` / `LucasKANConvNDLayer`: Uses Lucas polynomials.
    *   `GRAMKANLayer` / `GRAMKANConvNDLayer`: Uses Gram polynomials.
*   **Wavelet-Based:**
    *   `WavKANLayer` / `WavKANConvNDLayer`: Uses various wavelet functions (Mexican Hat, Morlet, DoG, Meyer, Shannon).
*   **Other Bases:**
    *   `FourierKANLayer` / `FourierKANConvNDLayer`: Uses Fourier series.
    *   `ReLUKANLayer` / `ReLUKANConvNDLayer`: Uses combinations of ReLU functions.

*   **Convolutional Variants:** 1D, 2D, and 3D convolutional versions are provided for most KAN types (`*KANConv1DLayer`, `*KANConv2DLayer`, `*KANConv3DLayer`).
*   **Convolution Factory:** A convenient `CONV_KAN_FACTORY` in `kan_conv.py` allows creating different convolutional layers (including standard `nn.Conv2d`) using string identifiers.
*   **Regularization:** Basic L1 regularization wrapper available (`utils.regularization.L1`).

## Available Layers

The following KAN layer types (both linear and convolutional variants) are implemented:

*   KAN (B-Spline)
*   FastKAN (RBF)
*   TaylorKAN
*   ChebyKAN (Chebyshev Polynomials)
*   LegendreKAN
*   LaguerreKAN
*   HermiteKAN
*   JacobiKAN
*   GegenbauerKAN
*   BersnsteinKAN (Bernstein Polynomials) (**Currently has a bug that make some of the model not learning**)
*   BesselKAN
*   FibonacciKAN
*   LucasKAN
*   GRAMKAN (Gram Polynomials)
*   WavKAN (Wavelets: Mexican Hat, Morlet, DoG, Meyer, Shannon)
*   FourierKAN
*   ReLUKAN

Please refer to the individual source files (e.g., `kan_layers.py`, `legendre_kan_layers.py`, `kan_conv.py`) for specific class names, parameters, and implementation details.

## Acknowledgements

The KAN layers and Convolutional KAN implementations in this folder adapt and build upon code from several excellent open-source projects and papers, including:

*   [TorchConv KAN](https://github.com/IvanDrokin/torch-conv-kan)
*   [OrthoPolyKANs](https://github.com/Boris-73-TA/OrthogPolyKANs)
*   [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN)
*   [Wav-KAN](https://github.com/zavareh1/Wav-KAN)
*   [efficient-kan](https://github.com/Blealtan/efficient-kan)
*   [GRAMKAN](https://github.com/Khochawongwat/GRAMKAN)
*   [FourierKAN](https://github.com/GistNoesis/FourierKAN)
*   [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN)

We thank the authors of these repositories and papers for their valuable contributions to the field and for making their work publicly available.
