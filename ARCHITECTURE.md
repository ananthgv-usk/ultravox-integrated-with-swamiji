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
To achieve the full persona, the **Base Ultravox Model** must be replaced with a **Fine-tuned Version**.

```mermaid
graph TD
    Dataset[("Custom Dataset<br/>(Books, Lectures, Q&A)")] 
    
    subgraph Training Phase
        Base[Ultravox v0.4 Base] --> Trainer
        Dataset --> Trainer
        Trainer -->|Fine-tuning| FT_Model{Ultravox Swamiji Model}
    end
    
    style FT_Model fill:#ffff99,stroke:#ff6600,stroke-width:4px
    
    subgraph Inference Pipeline Phase 3
        Input[Voice Input] --> FT_Model
        FT_Model -->|Context-Aware Text| TTS[Orpheus TTS]
        TTS --> Output[Final Voice Output]
    end
```

### Integration Steps
1.  **Prepare Data**: Format existing transcripts into `(audio, text)` or `(text instruction, text response)` pairs appropriate for Ultravox training.
2.  **Fine-tune**: Run Ultravox training on the A6000 pod.
3.  **Plug-in**: Update `ultravox_handler.py` to point to the new `checkpoint` instead of `fixie-ai/ultravox-v0_4`.
