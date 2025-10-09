# Error Solutions & Troubleshooting

## Main Errors and Solutions

### 1. Model Save Failure After Training

Problem: Model saved message appears but file not created

Cause:
- Duplicate trainer.train() calls
- Conflicting save logic

Solution:
- Clean up model.py train() function
- Call trainer.train() only once
- Place save logic in one clear location

---

### 2. ELECTRA Tensor Contiguous Error

Problem:

    ValueError: non contiguous tensor: electra.embeddings_project.weight

Cause: ELECTRA tensors stored non-contiguously in memory

Solution:

    # Add to TrainingArguments
    save_safetensors=False
    
    # Or make tensors contiguous before saving
    for param in model.parameters():
        param.data = param.data.contiguous()

---

### 3. Missing Tokenizer in Checkpoint

Problem: OSError: Can't load tokenizer

Cause: Trainer doesn't save tokenizer with checkpoint

Solution:

    # Manually copy tokenizer files
    cp models/final_model/tokenizer* models/checkpoint-XXXX/

---

### 4. Vocab Size Mismatch

Problem:

    IndexError: index out of range
    CUDA error: device-side assert triggered

Cause:
- Added 17 special tokens → tokenizer size 30017
- Model vocab_size remains 30000

Solution:

    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

---

### 5. CPU vs GPU Prediction Error

Problem: CUDA assert errors on GPU

Solution:

    # Use CPU temporarily
    self.device = torch.device('cpu')
    
    # Or debug mode
    CUDA_LAUNCH_BLOCKING=1 python src/ensemble.py

---

### 6. Missing torch Import

Problem: UnboundLocalError: torch not defined

Solution:

    # Add to top of model.py
    import torch

---

## Prevention Tips

### Verify Saves

    import os
    assert os.path.exists("models/final_model/model.safetensors")

### Check Vocab Size

    print(f"Tokenizer: {len(tokenizer)}")
    print(f"Model vocab: {model.config.vocab_size}")

### Use Checkpoints

    save_strategy="epoch"
    save_total_limit=2

---

## Debugging Commands

### Clear GPU Memory

    python -c "import torch; torch.cuda.empty_cache()"

### Verify Model Structure

    from transformers import AutoModel
    model = AutoModel.from_pretrained("path")
    print(model)

---

## References

- Transformers Troubleshooting
- PyTorch CUDA Errors Documentation
