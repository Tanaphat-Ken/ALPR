#!/usr/bin/env python3
"""Test the fix for province token embedding size."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from train.province_mapping import get_special_tokens
import torch

def test_fix():
    """Test the embedding resize fix."""
    print("=== Testing Province Token Fix ===")
    
    # Load base model
    model_id = "kkatiz/thai-trocr-thaigov-v2"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    
    print(f"Original embedding size: {model.decoder.get_input_embeddings().num_embeddings}")
    print(f"Original tokenizer len: {len(processor.tokenizer)}")
    
    # Add province tokens
    province_tokens = get_special_tokens("code")
    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": province_tokens})
    print(f"Added {added} province tokens")
    
    # Apply the fix
    new_vocab_size = len(processor.tokenizer)
    vocab = processor.tokenizer.get_vocab()
    max_token_id = max(vocab.values()) if vocab else 0
    required_vocab_size = max(new_vocab_size, max_token_id + 1)
    
    print(f"Tokenizer len: {new_vocab_size}")
    print(f"Max token ID: {max_token_id}")
    print(f"Required vocab size: {required_vocab_size}")
    
    # Resize embeddings
    model.decoder.resize_token_embeddings(required_vocab_size)
    model.config.decoder.vocab_size = required_vocab_size
    
    print(f"New embedding size: {model.decoder.get_input_embeddings().num_embeddings}")
    
    # Test with a province token
    dummy_image = torch.randn(1, 3, 384, 384)
    province_token_id = vocab[province_tokens[0]]  # <prov>
    last_province_id = max([vocab[t] for t in province_tokens])
    
    print(f"Testing with province token ID: {province_token_id}")
    print(f"Last province token ID: {last_province_id}")
    print(f"Embedding bounds: 0 - {model.decoder.get_input_embeddings().num_embeddings - 1}")
    
    # Check if all token IDs are within bounds
    invalid_ids = [id for id in vocab.values() if id >= model.decoder.get_input_embeddings().num_embeddings]
    if invalid_ids:
        print(f"ERROR: Invalid token IDs: {invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}")
        return False
    else:
        print("SUCCESS: All token IDs are within embedding bounds")
    
    # Test forward pass
    try:
        input_ids = torch.tensor([[0, province_token_id, last_province_id]], dtype=torch.long)
        outputs = model(
            pixel_values=dummy_image,
            decoder_input_ids=input_ids
        )
        print(f"Forward pass successful! Output shape: {outputs.logits.shape}")
        return True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fix()
    exit(0 if success else 1)