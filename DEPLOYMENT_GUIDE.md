# RunPod Deployment Guide

This guide documents how to deploy the `speech_pipeline` (Phase 2 with Ultravox & Orpheus) on a RunPod instance (e.g., RTX A6000).

## 1. Environment Setup

**Recommended Pod Template**: RunPod PyTorch 2.4.0 (CUDA 12.4.1)
**Requirements**: Python 3.10+ (System Python 3.11 is verified).

## 2. Transfer Files
Copy the `speech_pipeline` directory to the pod:
```bash
scp -r speech_pipeline root@<POD_IP>:/workspace/
```

## 3. Install Dependencies
SSH into the pod and install the requirements:
```bash
ssh root@<POD_IP>
cd /workspace/speech_pipeline
pip install -r requirements.txt
```
*Note: If `blinker` error occurs: `pip install --ignore-installed blinker`.*

## 4. Run Verification
Run the pipeline using the Hugging Face checkpoints:
```bash
python3 main.py \
  --input_audio arjuna_krishna.wav \
  --ultravox_model fixie-ai/ultravox-v0_4 \
  --tts_checkpoint unsloth/orpheus-3b-0.1-ft
```

## 5. Output
The script generates `final_output.wav` in the directory. Download it to listen:
```bash
scp root@<POD_IP>:/workspace/speech_pipeline/final_output.wav ./
```
