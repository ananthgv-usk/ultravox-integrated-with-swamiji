import whisper
import os

class STTHandler:
    def __init__(self, model_name="base", device="cpu"):
        print(f"Loading Whisper model: {model_name} on {device}...")
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing {audio_path}...")
        result = self.model.transcribe(audio_path)
        text = result["text"].strip()
        print(f"Transcription: {text}")
        return text

if __name__ == "__main__":
    # Test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()
    
    stt = STTHandler()
    print(stt.transcribe(args.audio))
