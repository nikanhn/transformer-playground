# Transformer Playground

This repository contains a simple implementation of a Transformer model in PyTorch. The code is structured to allow for easy experimentation with different model architectures and training configurations.

## Features

*   **Encoder and Decoder Implementation**: Includes implementations of both the Transformer encoder and decoder.
*   **Command-Line Interface**: A simple command-line interface is provided to run different experiments.
*   **Architectural Exploration**: The code is designed to be easily modified to explore different model architectures.

## How to Use

To run the different experiments, use the following commands:

*   **Encoder Task**: `python main.py --task encoder`
*   **Decoder Task**: `python main.py --task decoder`
*   **Architectural Exploration 1**: `python main.py --task exploration --subtask architectural1`
*   **Architectural Exploration 2**: `python main.py --task exploration --subtask architectural2`
*   **Performance Improvement**: `python main.py --task exploration --subtask performance`

## Requirements

*   Python 3.x
*   PyTorch
*   matplotlib
