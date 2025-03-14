"""
Modify CSM code to use local models for offline operation
This script patches the necessary files to use locally downloaded models instead of
downloading from HuggingFace.
"""

import os
import re
import shutil
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("code_modifier")

def backup_file(file_path):
    """Create a backup of the file"""
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    return backup_path

def patch_generator_py(local_config):
    """Patch generator.py to use local models"""
    file_path = "generator.py"
    backup_file(file_path)
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Patch import section to import local config
    if "import local_config" not in content:
        content = content.replace(
            "from huggingface_hub import hf_hub_download",
            "from huggingface_hub import hf_hub_download\n# Import local model configuration\nimport local_config"
        )
    
    # Patch CSM model loading to use local path
    content = re.sub(
        r"mimi_weight = hf_hub_download\(loaders\.DEFAULT_REPO, loaders\.MIMI_NAME\)",
        "mimi_weight = local_config.MODELS['mimi'] if hasattr(local_config, 'MODELS') else hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)",
        content
    )
    
    # Patch the load_csm_1b function to use local path
    pattern = r"def load_csm_1b\(ckpt_path: str = \"ckpt\.pt\", device: str = \"cuda\"\) -> Generator:"
    replacement = "def load_csm_1b(ckpt_path: str = None, device: str = \"cuda\") -> Generator:\n    # Use local path if available\n    if ckpt_path is None:\n        ckpt_path = local_config.MODELS['csm'] if hasattr(local_config, 'MODELS') else \"ckpt.pt\""
    
    content = re.sub(pattern, replacement, content)
    
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Patched {file_path} to use local models")

def patch_llm_py(local_config):
    """Patch llm.py to use local models"""
    file_path = "llm.py"
    backup_file(file_path)
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Add import for local config
    if "import local_config" not in content:
        import_statement = "import logging\n\n# Import local model configuration\nimport local_config"
        content = content.replace("import logging", import_statement)
    
    # Modify _find_available_model method to prioritize local model
    find_model_pattern = r"def _find_available_model\(self, model_name: Optional\[str\]\) -> str:.*?if model_name is not None:.*?return model_name.*?\n\s+# Try the default models"
    find_model_replacement = """def _find_available_model(self, model_name: Optional[str]) -> str:
        \"\"\"Find an available LLM model to use.\"\"\"
        # If model name is specified, use it
        if model_name is not None:
            return model_name
            
        # Check if we have a local model
        if hasattr(local_config, 'MODELS') and 'llm' in local_config.MODELS:
            logger.info(f"Using local LLM model: {local_config.MODELS['llm']}")
            return local_config.MODELS['llm']
            
        # Try the default models"""
    
    content = re.sub(find_model_pattern, find_model_replacement, content, flags=re.DOTALL)
    
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Patched {file_path} to use local models")

def patch_asr_py(local_config):
    """Patch asr.py to use local models"""
    file_path = "asr.py"
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found, skipping")
        return
        
    backup_file(file_path)
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Add import for local config
    if "import local_config" not in content:
        # Find the import section
        import_section = "import torch\nimport numpy as np\nfrom transformers import pipeline"
        new_import_section = "import torch\nimport numpy as np\nfrom transformers import pipeline\n\n# Import local model configuration\nimport local_config"
        
        content = content.replace(import_section, new_import_section)
    
    # Modify ASR initialization to use local model
    asr_init_pattern = r"self\.asr = pipeline\(\n\s+\"automatic-speech-recognition\",\n\s+model=model_name,\n\s+device=device\n\s+\)"
    asr_init_replacement = """self.asr = pipeline(
            "automatic-speech-recognition",
            model=local_config.MODELS['asr'] if hasattr(local_config, 'MODELS') and 'asr' in local_config.MODELS else model_name,
            device=device
        )"""
    
    content = re.sub(asr_init_pattern, asr_init_replacement, content)
    
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Patched {file_path} to use local models")

def patch_speech_interface_py(local_config):
    """Patch speech_interface.py to use local models"""
    file_path = "speech_interface.py"
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found, skipping")
        return
        
    backup_file(file_path)
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Add import for local config
    if "import local_config" not in content:
        import_statement = "from huggingface_hub import hf_hub_download\n\n# Import local model configuration\nimport local_config"
        content = content.replace("from huggingface_hub import hf_hub_download", import_statement)
    
    # Modify model loading to use local paths
    content = re.sub(
        r"csm_model_path = hf_hub_download\(repo_id=\"sesame/csm-1b\", filename=\"ckpt\.pt\"\)",
        "csm_model_path = local_config.MODELS['csm'] if hasattr(local_config, 'MODELS') else hf_hub_download(repo_id=\"sesame/csm-1b\", filename=\"ckpt.pt\")",
        content
    )
    
    with open(file_path, "w") as f:
        f.write(content)
    
    logger.info(f"Patched {file_path} to use local models")

def create_offline_launcher():
    """Create a launcher script for offline mode"""
    content = """#!/usr/bin/env python
"""\"
Offline launcher script for CSM speech interface.
This script runs the CSM system using locally downloaded models.
\"""

import os
import sys
import torch
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("offline_launcher")

# Configure PyTorch for Windows compatibility
os.environ["TORCH_COMPILE_BACKEND"] = "eager"  # Use eager mode instead of inductor

# Tell PyTorch to suppress errors and fall back to eager mode
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

# Disable cuda graph capture which can cause issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():
    parser = argparse.ArgumentParser(description="Run CSM in offline mode")
    parser.add_argument("--voice", type=str, default=None, 
                    help="Voice profile to use (default: first available)")
    parser.add_argument("--debug", action="store_true", 
                    help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print system info
    logger.info("Starting CSM in offline mode")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Check if local config exists
    try:
        import local_config
        logger.info("Local models configuration found")
    except ImportError:
        logger.error("Local configuration not found. Please run download_models.py first.")
        sys.exit(1)
    
    # Import speech interface
    from speech_interface import SpeechInterface
    
    # Create interface
    interface = SpeechInterface(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Set voice if specified
    if args.voice:
        try:
            interface.set_active_voice(args.voice)
            logger.info(f"Using voice profile: {args.voice}")
        except ValueError:
            available_voices = interface.list_voice_profiles()
            logger.warning(f"Voice profile '{args.voice}' not found. Available voices: {', '.join(available_voices)}")
            if available_voices:
                interface.set_active_voice(available_voices[0])
                logger.info(f"Using default voice profile: {available_voices[0]}")
    
    # Start conversation loop
    try:
        interface.conversation_loop()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Error in conversation loop: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
    
    with open("run_csm_offline.py", "w") as f:
        f.write(content)
    
    # Make the file executable on Unix systems
    if os.name != 'nt':  # not Windows
        os.chmod("run_csm_offline.py", 0o755)
    
    # Create batch file for Windows
    with open("run_csm_offline.bat", "w") as f:
        f.write("@echo off\n")
        f.write("echo Starting CSM in offline mode...\n")
        f.write("python run_csm_offline.py %*\n")
    
    logger.info("Created offline launcher scripts: run_csm_offline.py and run_csm_offline.bat")

def main():
    # Check if local_config.py exists
    if not os.path.exists("local_models/local_config.py"):
        logger.error("Local configuration not found. Please run download_models.py first.")
        sys.exit(1)
    
    # Import the local config
    sys.path.append("local_models")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_config", "local_models/local_config.py")
        local_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_config)
    except ImportError:
        logger.error("Could not import local_config.py. Please run download_models.py first.")
        sys.exit(1)
    
    # Copy the config file to the main directory
    shutil.copy2("local_models/local_config.py", "local_config.py")
    logger.info("Copied local_config.py to the main directory")
    
    # Patch the files
    patch_generator_py(local_config)
    patch_llm_py(local_config)
    patch_asr_py(local_config)
    patch_speech_interface_py(local_config)
    
    # Create offline launcher
    create_offline_launcher()
    
    logger.info("All modifications complete!")
    logger.info("To run CSM in offline mode, use: python run_csm_offline.py")
    
if __name__ == "__main__":
    main()
