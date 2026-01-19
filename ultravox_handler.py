import transformers
import numpy as np
import librosa
import torch
import os

import transformers
import numpy as np
import librosa
import torch
import os

class UltravoxHandler:
    def __init__(self, model_id="fixie-ai/ultravox-v0_4", device="cuda"):
        self.device = device
        print(f"Loading Ultravox model: {model_id} on {device}...")
        
        # Check for CUDA/MPS
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            print(f"Device adjusted to: {self.device}")

        # Load Processor and Model explicitly with trust_remote_code=True
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            raise e
        
    def process(self, audio_path, prompt="Transcribe this audio."):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        print(f"Processing {audio_path} with Ultravox...")
        
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
        
        turns = [
            {
                "role": "user",
                "content": f"<|audio|> {prompt}"
            }
        ]
        
        # Prepare inputs using the processor
        text = self.processor.tokenizer.apply_chat_template(turns, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=audio, return_tensors="pt", sampling_rate=16000)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=True, # Optional: set to False if you want deterministic structure
                temperature=0.6
            )
        
        # Decode: skip input tokens
        input_len = inputs["input_ids"].shape[1]
        output_text = self.processor.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        
        print(f"Ultravox Output: {output_text}")
        return output_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, default="fixie-ai/ultravox-v0_4")
    args = parser.parse_args()
    
    handler = UltravoxHandler(model_id=args.model)
    print(handler.process(args.audio))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--model", type=str, default="fixie-ai/ultravox-v0.4")
    args = parser.parse_args()
    
    handler = UltravoxHandler(model_id=args.model)
    print(handler.process(args.audio))
