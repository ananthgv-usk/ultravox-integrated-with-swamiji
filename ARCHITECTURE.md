# System Architecture

## Current Verification State (Phase 2)
The current pipeline uses a modular approach where the "Brain" and "Voice" are separate.

### Verification Data
*   **Input**: `speech_pipeline/arjuna_krishna.wav` (Voice Note asking about Arjuna & Krishna)
*   **Ultravox Output**: "You're referring to the relationship between Arjuna and Lord Krishna."
*   **TTS Output**: `speech_pipeline/final_output_full.wav` (Generated Speech)

### System Diagram
```mermaid
graph TD
    User((User)) -->|Voice Input| A[Input Audio]
    
    subgraph Phase 2: Current Pipeline
        A --> B(Ultravox v0.4)
        B --- noteB["Generic Llama 3 Brain<br/>(Base Model)"]
        
        B -->|Text Response| C[Text Output]
        
        C --> D(Orpheus TTS)
        D --- noteD["Swami Nithyananda Voice<br/>(Checkpoint 450)"]
        
        D -->|Audio Validated| E[Output Audio]
    end
    
    E --> User
    
    style B fill:#ff9999,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style D fill:#99ff99,stroke:#333,stroke-width:2px
```

## Phase 3: Integration Required
To achieve the full persona, we must execute the following **TODOs** to replace the Generic Brain.

```mermaid
graph TD
    Dataset[("Custom Dataset<br/>(Books, Lectures, Q&A)<br/><b>[TODO: Prepare JSONL]</b>")] 
    
    subgraph "Training Phase [TODO: Run on RunPod]"
        Base[Ultravox v0.4 Base] --> Trainer[Ultravox Trainer]
        Dataset --> Trainer
        Trainer -->|Fine-tuning| FT_Model{Ultravox Swamiji Model}
    end
    
    style Dataset fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style Trainer fill:#ffcccc,stroke:#ff0000,stroke-width:2px
    style FT_Model fill:#ffff99,stroke:#ff6600,stroke-width:4px
    
    subgraph Inference Pipeline Phase 3
        Input[Voice Input] --> FT_Model
        FT_Model -->|Context-Aware Text| TTS[Orpheus TTS]
        TTS --> Output[Final Voice Output]
    end
```

### Integration Steps (Detailed)
1.  **[TODO] Data Prep**: Convert your corpus into Ultravox-compatible `jsonl` format containing `{"audio": ..., "text": ...}` (or text-only instruction tuning).
2.  **[TODO] Training**: Use the A6000 GPU to fine-tune `fixie-ai/ultravox-v0_4` on this dataset.
3.  **[TODO] Inference Switch**: Update `ultravox_handler.py` to point to the new local checkpoint.
