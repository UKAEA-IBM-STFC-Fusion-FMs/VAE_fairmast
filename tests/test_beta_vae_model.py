"""
Test wrapper for beta_vae_model.py
"""
import sys
import subprocess
from pathlib import Path

def main():
    """Execute the beta VAE model script as a test."""
    project_root = Path(__file__).parent.parent.resolve()
    
    # Path to the actual model script
    model_script = project_root / "scripts" / "pipelines" / "models" / "beta_vae_model.py"
    
    if not model_script.exists():
        print(f"Model script not found: {model_script}")
        sys.exit(1)
    
    print(f"Executing beta VAE model script: {model_script}")
    print(f"Working directory: {project_root}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(model_script)],
            cwd=str(project_root),  # Important: run from project root
            check=True  # Raise exception if the script fails
        )
        
        print("Beta VAE model script completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"Beta VAE model script failed with return code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error executing beta VAE model script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
