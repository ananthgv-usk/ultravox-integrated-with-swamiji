import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
import soundfile as sf
import os
import numpy as np

# Special tokens (matching reference)
START_OF_HUMAN = 128259
END_OF_TEXT = 128009
END_OF_HUMAN = 128260
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
AUDIO_TOKENS_START = 128266

class TTSHandler:
    def __init__(self, checkpoint_path="unsloth/orpheus-3b-0.1-ft", device="cuda"):
        self.device = device
        print(f"Using device: {self.device}")
        
        print(f"Loading tokenizer from {checkpoint_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        print(f"Loading Model from {checkpoint_path}...")
        # Use sdpa for speed if on cuda
        attn = "sdpa" if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else "eager"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device,
            attn_implementation=attn
        )
        self.model.eval()
        
        print(f"Loading SNAC model (24kHz)...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

    def redistribute_codes(self, code_list):
        """Redistribute 7-token pattern back to 3 SNAC layers (24kHz)"""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        num_frames = len(code_list) // 7
        
        for i in range(num_frames):
            base = 7 * i
            # L0
            layer_1.append(code_list[base])
            # L1[0]
            layer_2.append(code_list[base + 1] - 4096)
            # L2[0]
            layer_3.append(code_list[base + 2] - (2 * 4096))
            # L2[1]
            layer_3.append(code_list[base + 3] - (3 * 4096))
            # L1[1]
            layer_2.append(code_list[base + 4] - (4 * 4096))
            # L2[2]
            layer_3.append(code_list[base + 5] - (5 * 4096))
            # L2[3]
            layer_3.append(code_list[base + 6] - (6 * 4096))
        
        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(self.device),
            torch.tensor(layer_2).unsqueeze(0).to(self.device),
            torch.tensor(layer_3).unsqueeze(0).to(self.device)
        ]
        return codes

    def synthesize(self, text, output_path):
        print(f"Synthesizing text: {text}")
        
        # Prepare Input
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
        
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        input_ids_dev = modified_input_ids.to(self.device)
        attention_mask_dev = torch.ones_like(input_ids_dev).to(self.device)
        
        # Auto-calculate max_tokens (approx 80 per word)
        word_count = len(text.split())
        max_tokens = max(int(word_count * 80), 1000)
        
        print(f"Generating audio tokens (max {max_tokens})...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids_dev,
                attention_mask=attention_mask_dev,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=END_OF_SPEECH
            )
            
        # Extract audio codes
        print("Extracting audio codes...")
        token_indices = (generated_ids == START_OF_SPEECH).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_idx = token_indices[1][-1].item()
            cropped = generated_ids[:, last_idx + 1:]
        else:
            print("Warning: START_OF_SPEECH not found, utilizing full output")
            cropped = generated_ids
            
        # Remove EOS and flatten
        raw_codes = []
        for t in cropped[0]:
            if t == END_OF_SPEECH:
                break
            raw_codes.append(t.item())
            
        # Normalize to AUDIO_TOKENS_START
        subtracted_codes = []
        for c in raw_codes:
            if c >= AUDIO_TOKENS_START:
                subtracted_codes.append(c - AUDIO_TOKENS_START)
        
        # Trim to multiple of 7
        new_length = (len(subtracted_codes) // 7) * 7
        subtracted_codes = subtracted_codes[:new_length]
        
        if len(subtracted_codes) > 0:
            codes_tensors = self.redistribute_codes(subtracted_codes)
            
            with torch.no_grad():
                audio_hat = self.snac_model.decode(codes_tensors)
                
            audio_np = audio_hat.detach().squeeze().cpu().numpy()
            sf.write(output_path, audio_np, 24000)
            print(f"Saved {output_path}")
            return output_path
        else:
             print("No valid audio codes generated.")
             return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--checkpoint", type=str, default="unsloth/orpheus-3b-0.1-ft")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        if torch.backends.mps.is_available(): device = 'mps'

    tts = TTSHandler(checkpoint_path=args.checkpoint, device=device)
    tts.synthesize(args.text, args.output)
