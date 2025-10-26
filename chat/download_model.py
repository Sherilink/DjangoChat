import os
import requests

model_url = "https://drive.google.com/uc?export=download&id=1lgmgKLTSleYr_rR8nhClUhuQ1WOiCFQx"
model_path = os.path.join(os.path.dirname(__file__), "models/phi-2.Q4_K_M.gguf")

os.makedirs(os.path.dirname(model_path), exist_ok=True)

if not os.path.exists(model_path):
    print("Downloading model...")
    r = requests.get(model_url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded.")
else:
    print("Model already exists.")


