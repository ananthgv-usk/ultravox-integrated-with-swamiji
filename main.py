import os
import argparse
from ultravox_handler import UltravoxHandler
from tts_handler import TTSHandler
import torch

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Speech Pipeline (Phase 2: Ultravox)")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to input voice note")
    parser.add_argument("--output_audio", type=str, default="final_output.wav", help="Path to output audio")
    parser.add_argument("--ultravox_model", type=str, default="fixie-ai/ultravox-v0_4", help="Ultravox model ID")
    parser.add_argument("--tts_checkpoint", type=str, default="/workspace/orpheus_947_final", help="Path to Orpheus checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    # Device setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. Switching to CPU.")
        device = 'cpu'
        if torch.backends.mps.is_available():
             print("MPS available. Switching to MPS.")
             device = 'mps'

    # 1. Audio Intelligence (Ultravox)
    print("--- Step 1: Audio Intelligence (Ultravox) ---")
    ultravox = UltravoxHandler(model_id=args.ultravox_model, device=device)
    # Using a conversational prompt
    generated_text = ultravox.process(args.input_audio, prompt="Reply with a very short sentence (max 10 words).")
    
    if not generated_text:
        print("Error: No text generated.")
        return

    # 2. Text to Speech
    print(f"\n--- Step 2: Text to Speech (Input: {len(generated_text)} chars) ---")
    try:
        tts = TTSHandler(checkpoint_path=args.tts_checkpoint, device=device)
        tts.synthesize(generated_text, args.output_audio)
    except Exception as e:
        print(f"Error in TTS step: {e}")

if __name__ == "__main__":
    main()
