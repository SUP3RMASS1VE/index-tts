
# IndexTTS Web UI
![Screenshot 2025-04-17 233756](https://github.com/user-attachments/assets/c6d89219-29b2-4994-a1fb-69f2431e93bd)

A simple web interface for the [IndexTTS model](https://huggingface.co/IndexTeam/Index-TTS), an industrial-level controllable and efficient zero-shot text-to-speech system.

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
</p>

## Features

- Zero-shot voice cloning: Generate speech that matches a reference audio file
- Simple web interface using Gradio
- Automatic model downloading from Hugging Face
- Support for English and Chinese text

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. Install the requirements:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

2. Run the web UI:
```bash
python webui.py
```

On first run, the script will automatically download the necessary model files from the Hugging Face repository. This might take some time depending on your internet connection speed.

## Usage

1. Upload a reference audio file (voice you want to clone)
2. Enter the text you want to convert to speech
3. Click "Generate Speech"
4. Listen to the generated audio

## Model Information

IndexTTS is a GPT-style text-to-speech model mainly based on XTTS and Tortoise, with enhancements including:
- Correction of pronunciation for Chinese characters using pinyin
- Control of pauses at any position through punctuation
- Improved speaker condition feature representation
- Integration of BigVGAN2 for optimized audio quality

For more information, visit the [IndexTTS Hugging Face page](https://huggingface.co/IndexTeam/Index-TTS) or read the [research paper](https://arxiv.org/abs/2502.05512).

## Citation

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
