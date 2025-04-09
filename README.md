# Piper TTS Example

This project demonstrates how to use Piper TTS to generate speech from text.

## Prerequisites

*   Python 3.11
*   pip
*   git

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd tts
    ```

2.  Create a virtual environment:

    ```bash
    python3.11 -m venv venv-py311
    source venv-py311/bin/activate
    ```

3.  Install the dependencies:

    ```bash
    pip install piper-tts==1.2.0
    ```

4.  Download the model files:

    ```bash
    cd modelos
    huggingface-cli download rhasspy/piper-voices es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx --local-dir . --local-dir-use-symlinks False
    huggingface-cli download rhasspy/piper-voices es/es_ES/carlfm/x_low/es_ES-carlfm-x_low.onnx.json --local-dir . --local-dir-use-symlinks False
    cd ..
    ```

## Usage

1.  Run the script:

    ```bash
    python decir_hola.py
    ```

## Troubleshooting

*   If you encounter a `404 Not Found` error when downloading the model files, make sure you have the `huggingface-cli` installed and that you are using the correct paths to the model files.
*   If you encounter an error related to `sample_width` or `num_channels`, make sure you are using the correct version of the `decir_hola.py` script.

## Repository Structure

```
.
├── decir_hola.py
├── modelos
│   ├── es
│   │   └── es_ES
│   │       └── carlfm
│   │           └── x_low
│   │               ├── es_ES-carlfm-x_low.onnx
│   │               └── es_ES-carlfm-x_low.onnx.json
├── venv-py311
└── README.md