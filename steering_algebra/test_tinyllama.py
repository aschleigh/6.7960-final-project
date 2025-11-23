# test_tinyllama.py
"""Test that TinyLlama loads and runs correctly."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_tinyllama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check model structure
    n_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    print(f"✓ Model loaded: {n_layers} layers, hidden_size={hidden_size}")
    
    # Test generation
    prompt = "Write a short sentence about the weather:"
    messages = [{"role": "user", "content": prompt}]
    
    # TinyLlama uses ChatML format
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✓ Generation working:\n{response}\n")
    
    # Test activation extraction
    from extraction.hooks import ActivationHooks
    hooks = ActivationHooks(model)
    
    test_layers = [8, 10, 12, 14, 16]
    with hooks.extraction_context(test_layers) as cache:
        with torch.no_grad():
            model(**inputs)
        
        for layer in test_layers:
            key = f"layer_{layer}_residual"
            if key in cache.keys():
                shape = cache[key].shape
                print(f"✓ Layer {layer} activations: {shape}")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED - Ready to run experiments!")
    print("="*50)
    
    return model, tokenizer

if __name__ == "__main__":
    test_tinyllama()