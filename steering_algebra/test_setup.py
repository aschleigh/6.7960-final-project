# test_setup.py
"""Quick test to verify setup works."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_setup():
    print("Testing setup...")
    
    # Test imports
    from config import cfg
    from data.contrastive_pairs import get_contrastive_pairs
    from data.prompts import get_test_prompts
    print("✓ Imports working")
    
    # Test contrastive pairs
    pairs = get_contrastive_pairs("formal", n_pairs=5)
    assert len(pairs) == 5
    assert all(len(p) == 2 for p in pairs)
    print("✓ Contrastive pairs working")
    
    # Test prompts
    prompts = get_test_prompts()
    assert len(prompts) > 0
    print("✓ Test prompts working")
    
    # Test model loading (use small model for test)
    print("Loading small model for test...")
    model_name = "gpt2"  # Small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✓ Model loading working")
    
    # Test hooks
    from extraction.hooks import ActivationHooks
    hooks = ActivationHooks(model)
    print("✓ Hooks working")
    
    # Test generation
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    text = tokenizer.decode(outputs[0])
    print(f"✓ Generation working: '{text[:50]}...'")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
    print("\nYou're ready to run the full pipeline.")
    # print("Run: python run_week1.py --model meta-llama/Meta-Llama-3-8B-Instruct")
    print("Run: python run_week1.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    

if __name__ == "__main__":
    test_setup()