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
import io
import random

# Custom stderr filter to hide specific CUDA errors
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.blacklist = [
            "LINK : fatal error LNK1181: cannot open input file 'aio.lib'",
            "LINK : fatal error LNK1181: cannot open input file 'cufile.lib'",
            "NOTE: Redirects are currently not supported in Windows or MacOs",
            "test.c"
        ]
    
    def write(self, message):
        if not any(error in message for error in self.blacklist):
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
        
    def isatty(self):
        return self.original_stderr.isatty()
        
    def fileno(self):
        return self.original_stderr.fileno()

# Install our custom filter
sys.stderr = FilteredStderr(sys.stderr)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # Added to suppress FutureWarnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning'

# Monkey patch torch._pytree to silence FutureWarning messages
import torch.utils._pytree
if hasattr(torch.utils._pytree, "_register_pytree_node"):
    original_register = torch.utils._pytree._register_pytree_node
    def patched_register(*args, **kwargs):
        # Suppress the warning by using the new API if available
        if hasattr(torch.utils._pytree, "register_pytree_node"):
            return torch.utils._pytree.register_pytree_node(*args, **kwargs)
        else:
            return original_register(*args, **kwargs)
    torch.utils._pytree._register_pytree_node = patched_register

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

# Add more specific CUDA warning suppression
if torch.cuda.is_available():
    # Suppress CUDA initialization messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # More controlled CUDA launches
    os.environ['TORCH_CUDA_ARCH_LIST'] = 'All'  # Avoid architecture-specific warnings
    
    # Suppress NVRTC warnings
    if hasattr(torch.cuda, 'nvrtc'):
        torch._C._jit_set_nvrtc_options(["--display-error-number", "--suppress-warnings"])

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
# More extensive suppression during model loading
try:
    # Completely silence all output during model loading
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Try loading the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = IndexTTS(
        model_dir="checkpoints", 
        cfg_path="checkpoints/config.yaml",
        device=device,
        use_cuda_kernel=False  # Disable CUDA kernel to avoid compilation issues
    )
except Exception as e:
    # Restore stderr and stdout before printing errors
    sys.stdout, sys.stderr = original_stdout, original_stderr
    print(f"Warning: Error during model initialization: {e}")
    print("Falling back to CPU mode")
    try:
        tts = IndexTTS(
            model_dir="checkpoints", 
            cfg_path="checkpoints/config.yaml",
            device="cpu",
            use_cuda_kernel=False
        )
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        sys.exit(1)
finally:
    # Restore stdout and stderr
    if 'original_stdout' in locals() and 'original_stderr' in locals():
        sys.stdout, sys.stderr = original_stdout, original_stderr

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def infer(voice, text, output_path=None, seed=None):
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    tts.infer(voice, text, output_path)
    return output_path

def gen_single(prompt, text, seed=None):
    # Use random seed if seed is 0 or None
    if seed is None or seed == 0:
        seed = None
    else:
        seed = int(seed)  # Ensure it's an integer
    
    output_path = infer(prompt, text, seed=seed)
    return gr.update(value=output_path, visible=True)

def update_prompt_audio():
    # Always keep button interactive
    return gr.update(interactive=True, variant="primary")

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
    --content-width: 1100px;
}

body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', sans-serif;
}

#main-container {
    width: var(--content-width) !important;
    max-width: 90% !important;
    margin: 0 auto !important;
}

.gradio-container {
    width: 100% !important;
    margin: 0 auto !important;
}

/* Tab and container styling */
.tab-nav {
    margin-bottom: 1.5rem !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.tabitem {
    padding: 1.5rem !important;
}

/* Fix audio elements */
.audio-player {
    width: 100% !important;
    height: auto !important;
    margin-bottom: 10px !important;
    border-radius: var(--border-radius) !important;
    overflow: hidden !important;
}

.audio-player .wrapper {
    width: 100% !important;
}

.audio-player .controls {
    padding: 10px !important;
}

/* Better column balancing */
.gr-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 20px !important;
    width: 100% !important;
    margin-bottom: 20px !important;
}

/* Button styling */
.gr-button {
    background-color: var(--primary-color) !important;
    border: none !important;
    color: white !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow) !important;
    width: 100% !important;
    height: 46px !important;
}

.gr-button:hover {
    background-color: var(--secondary-color) !important;
    transform: translateY(-2px) !important;
}

/* Card layout */
.card-title {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    margin-bottom: 1rem !important;
    color: var(--text-color) !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding-bottom: 0.5rem !important;
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
    width: 100% !important;
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

/* Fix slider appearance */
.gr-slider {
    width: 100% !important;
}

/* Textarea placeholder */
.gr-textarea::placeholder {
    color: var(--secondary-text-color) !important;
    opacity: 0.7 !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    font-size: 0.9rem;
    color: var(--secondary-text-color);
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=["Inter", "sans-serif"]), elem_id="main-container") as demo:
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
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                gr.HTML('<div class="card-title">Voice Reference</div>')
                os.makedirs("prompts", exist_ok=True)
                gr.HTML('<div style="margin-bottom: 10px; color: #94a3b8; font-size: 0.9rem;">For best results, use clean voice audio less than 1 minute long.</div>')
                prompt_audio = gr.Audio(label="Upload reference audio", 
                                      sources=["upload", "microphone"], 
                                      type="filepath",
                                      elem_id="audio_component")
                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]
            
            with gr.Column(scale=2, min_width=500):
                gr.HTML('<div class="card-title">Text Input</div>')
                input_text_single = gr.Textbox(
                    label="Enter text to convert to speech",
                    placeholder="Type your text here...",
                    lines=5
                )
                
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            gr.HTML('<div class="card-title">Seed</div>')
                        seed_input = gr.Slider(
                            minimum=0,
                            maximum=1000000,
                            step=1,
                            label="",
                            value=0,
                            info="Set to 0 for random seed"
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.HTML('<div style="height: 38px;"></div>')
                        gen_button = gr.Button("🔊 Generate Speech", variant="primary", size="lg")
        
        gr.HTML('<div class="card-title">Generated Output</div>')
        output_audio = gr.Audio(label="", visible=True, elem_id="output_audio")
    
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
                     inputs=[prompt_audio, input_text_single, seed_input],
                     outputs=[output_audio])


if __name__ == "__main__":
    # Disable Gradio's update message
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    try:
        # Ensure our custom filter is active
        if not isinstance(sys.stderr, FilteredStderr):
            sys.stderr = FilteredStderr(sys.stderr)
            
        # Disable the torch distributed elastic warning
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
        
        demo.queue(20)
        demo.launch(server_name="127.0.0.1", share=False)
    except Exception as e:
        print(f"Error launching the web interface: {e}")
        print("Try running with CPU only mode if you're having GPU-related issues")
