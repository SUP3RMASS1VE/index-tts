import os
import shutil
import sys
import threading
import time
import requests
import huggingface_hub
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import yaml
import warnings
import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::DeprecationWarning'

# Suppress CUDA compilation warnings by redirecting stderr temporarily during model loading
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

def download_model_files():
    """Download model files from Hugging Face on first run"""
    repo_id = "IndexTeam/Index-TTS"
    os.makedirs("checkpoints", exist_ok=True)
    
    # Check if config file exists, download if not
    config_path = os.path.join("checkpoints", "config.yaml")
    if not os.path.exists(config_path):
        print("Downloading config.yaml...")
        try:
            hf_hub_download(repo_id=repo_id, filename="config.yaml", local_dir="checkpoints", local_dir_use_symlinks=False)
        except Exception as e:
            print(f"Error downloading config.yaml: {e}")
            return False
    
    # Load config to get model filenames
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return False
    
    # Get model filenames from config
    model_files = [
        config.get("dvae_checkpoint", "dvae.pth"),
        config.get("gpt_checkpoint", "gpt.pth"),
        config.get("bigvgan_checkpoint", "bigvgan_generator.pth"),
        "bpe.model"  # Tokenizer model
    ]
    
    # Download each model file
    for file in model_files:
        file_path = os.path.join("checkpoints", file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            try:
                hf_hub_download(repo_id=repo_id, filename=file, local_dir="checkpoints", local_dir_use_symlinks=False)
                print(f"Successfully downloaded {file}")
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                return False
    
    print("All model files downloaded successfully!")
    return True

# Call the download function before importing the TTS model
if not all(os.path.exists(os.path.join("checkpoints", f)) for f in ["config.yaml", "dvae.pth", "gpt.pth", "bigvgan_generator.pth", "bpe.model"]):
    print("First run detected. Downloading model files...")
    if not download_model_files():
        print("Error downloading model files. Please check your internet connection and try again.")
        sys.exit(1)

import gradio as gr

try:
    from indextts.infer import IndexTTS
    from tools.i18n.i18n import I18nAuto
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Installing requirements...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    # Try importing again
    from indextts.infer import IndexTTS
    from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="en")  # Changed to English
MODE = 'local'

print("Loading IndexTTS model...")
with suppress_stdout_stderr():
    # Fix: Pass explicit device parameter and disable CUDA kernel for compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = IndexTTS(
        model_dir="checkpoints", 
        cfg_path="checkpoints/config.yaml",
        device=device,
        use_cuda_kernel=False  # Disable CUDA kernel to avoid compilation issues
    )

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def infer(voice, text, output_path=None):
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    tts.infer(voice, text, output_path)
    return output_path

def gen_single(prompt, text):
    output_path = infer(prompt, text)
    return gr.update(value=output_path, visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

# Custom CSS for better UI - Dark Theme
css = """
:root {
    --primary-color: #3b82f6;
    --secondary-color: #2563eb;
    --accent-color: #60a5fa;
    --background-color: #121212;
    --surface-color: #1e1e1e;
    --text-color: #e2e8f0;
    --secondary-text-color: #94a3b8;
    --border-color: #2d3748;
    --border-radius: 8px;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
}

.main {
    max-width: 1200px;
    margin: 0 auto;
}

.gradio-container {
    margin: 0 auto;
}

.header {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: white;
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.header p {
    margin-top: 0.5rem;
    opacity: 0.9;
    font-size: 1.1rem;
}

.gr-button {
    background-color: var(--primary-color) !important;
    border: none !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
}

.gr-button:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
}

.gr-button:disabled {
    opacity: 0.7 !important;
    cursor: not-allowed !important;
}

.gr-box, .gr-form, .gr-panel {
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow) !important;
    background-color: var(--surface-color) !important;
    border: 1px solid var(--border-color) !important;
}

.gr-padded {
    padding: 1.5rem !important;
}

.gr-input, .gr-textarea {
    border-radius: var(--border-radius) !important;
    border: 1px solid var(--border-color) !important;
    background-color: #2d3748 !important;
    color: var(--text-color) !important;
    padding: 0.75rem !important;
    transition: all 0.2s ease !important;
}

.gr-input:focus, .gr-textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

.gr-form > div {
    margin-bottom: 1.5rem !important;
}

.gr-label {
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    color: var(--text-color) !important;
}

.gr-tab {
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    color: var(--secondary-text-color) !important;
}

.gr-tab-selected {
    border-color: var(--primary-color) !important;
    color: var(--primary-color) !important;
}

/* Custom footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    font-size: 0.9rem;
    color: var(--secondary-text-color);
}

/* Card layout */
.card {
    background-color: var(--surface-color);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-color);
}

/* Dark theme specific adjustments */
.contain {
    background-color: var(--surface-color) !important;
}

.dark .gr-input, .dark .gr-textarea {
    background-color: #2d3748 !important;
    color: var(--text-color) !important;
}

.dark label, .dark .gr-label {
    color: var(--text-color) !important;
}

.dark .gr-checkbox {
    border-color: var(--border-color) !important;
}

.dark .gr-radio {
    border-color: var(--border-color) !important;
}

/* Audio input styling */
.audio-recorder {
    background-color: var(--surface-color) !important;
    border: 1px solid var(--border-color) !important;
}

.uploadButton {
    color: var(--text-color) !important;
    background-color: var(--surface-color) !important;
}

/* Textarea placeholder */
.gr-textarea::placeholder {
    color: var(--secondary-text-color) !important;
    opacity: 0.7 !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=["Inter", "sans-serif"])) as demo:
    mutex = threading.Lock()
    
    # Header
    gr.HTML('''
    <div class="header">
        <h1>IndexTTS</h1>
        <p>An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</p>
    </div>
    ''')
    
    # Main content
    with gr.Tab("Audio Generation"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="card-title">Voice Reference</div>')
                os.makedirs("prompts", exist_ok=True)
                prompt_audio = gr.Audio(label="Upload reference audio", key="prompt_audio",
                                      sources=["upload", "microphone"], type="filepath",
                                      elem_classes="gr-box gr-padded")
                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]
            
            with gr.Column(scale=2):
                gr.HTML('<div class="card-title">Text Input</div>')
                input_text_single = gr.Textbox(
                    label="Enter text to convert to speech",
                    placeholder="Type your text here...",
                    key="input_text_single",
                    lines=5,
                    elem_classes="gr-box gr-padded"
                )
                gen_button = gr.Button("🔊 Generate Speech", key="gen_button", interactive=True, size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.HTML('<div class="card-title">Generated Output</div>')
                output_audio = gr.Audio(label="Generated Result", visible=False, key="output_audio", elem_classes="gr-box gr-padded")
    
    # Footer
    gr.HTML('''
    <div class="footer">
        <p>IndexTTS - Powered by AI for high-quality text-to-speech generation</p>
    </div>
    ''')

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[prompt_audio, input_text_single],
                     outputs=[output_audio])


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="127.0.0.1", share=False)
