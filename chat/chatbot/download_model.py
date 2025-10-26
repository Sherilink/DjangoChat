import os
import requests

def download_model():
    """
    Downloads the model from Google Drive if it doesn't exist already.
    """
    model_url = "https://drive.google.com/uc?export=download&id=1lgmgKLTSleYr_rR8nhClUhuQ1WOiCFQx"
    model_path = os.path.join(os.path.dirname(__file__), "models/phi-2.Q4_K_M.gguf")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        print("Downloading model...")
        r = requests.get(model_url, stream=True)
        r.raise_for_status()  
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded.")
    else:
        print("Model already exists.")
if __name__ == "__main__":
    download_model()




