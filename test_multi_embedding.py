#!/usr/bin/env python3
"""
Test script to verify multi-embedding feature works correctly.
This script creates a small model and verifies that residual embeddings
are created at the correct layers.
"""

import torch
from torchtitan.models.llama3.model.args import TransformerModelArgs
from torchtitan.models.llama3.model.model import Transformer, TransformerBlock


def test_multi_embedding_disabled():
    """Test that no residual embeddings are created when disabled."""
    print("Test 1: Multi-embedding disabled")
    print("-" * 50)

    model_args = TransformerModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=1000,
        multi_embedding_enabled=False,
    )

    with torch.device("meta"):
        model = Transformer(model_args)

    residual_emb_count = 0
    for layer_id, layer in model.layers.items():
        if layer.residual_embedding is not None:
            residual_emb_count += 1
            print(f"  Layer {layer_id}: Has residual embedding (UNEXPECTED!)")

    if residual_emb_count == 0:
        print("✓ PASSED: No residual embeddings created")
    else:
        print(f"✗ FAILED: Found {residual_emb_count} residual embeddings")
    print()
    return residual_emb_count == 0


def test_multi_embedding_enabled():
    """Test that residual embeddings are created at correct intervals."""
    print("Test 2: Multi-embedding enabled (interval=4)")
    print("-" * 50)

    model_args = TransformerModelArgs(
        dim=512,
        n_layers=12,
        n_heads=8,
        vocab_size=1000,
        multi_embedding_enabled=True,
        multi_embedding_interval=4,
    )

    with torch.device("meta"):
        model = Transformer(model_args)

    expected_layers = {4, 8}  # Layers 4 and 8 should have residual embeddings
    actual_layers = set()

    for layer_id, layer in model.layers.items():
        layer_num = int(layer_id)
        if layer.residual_embedding is not None:
            actual_layers.add(layer_num)
            print(f"  Layer {layer_id}: Has residual embedding ✓")
        else:
            if layer_num in expected_layers:
                print(f"  Layer {layer_id}: Missing residual embedding ✗")

    if actual_layers == expected_layers:
        print("✓ PASSED: Residual embeddings at correct layers")
    else:
        print(f"✗ FAILED: Expected {expected_layers}, got {actual_layers}")
    print()
    return actual_layers == expected_layers


def test_multi_embedding_interval():
    """Test different intervals."""
    print("Test 3: Multi-embedding with interval=2")
    print("-" * 50)

    model_args = TransformerModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=1000,
        multi_embedding_enabled=True,
        multi_embedding_interval=2,
    )

    with torch.device("meta"):
        model = Transformer(model_args)

    expected_layers = {2, 4, 6}  # Layers 2, 4, 6 should have residual embeddings
    actual_layers = set()

    for layer_id, layer in model.layers.items():
        layer_num = int(layer_id)
        if layer.residual_embedding is not None:
            actual_layers.add(layer_num)
            print(f"  Layer {layer_id}: Has residual embedding ✓")

    if actual_layers == expected_layers:
        print("✓ PASSED: Residual embeddings at correct layers")
    else:
        print(f"✗ FAILED: Expected {expected_layers}, got {actual_layers}")
    print()
    return actual_layers == expected_layers


def test_forward_pass():
    """Test that forward pass works with multi-embedding."""
    print("Test 4: Forward pass with multi-embedding")
    print("-" * 50)

    model_args = TransformerModelArgs(
        dim=512,
        n_layers=8,
        n_heads=8,
        vocab_size=1000,
        multi_embedding_enabled=True,
        multi_embedding_interval=4,
        max_seq_len=128,
    )

    try:
        model = Transformer(model_args)
        model.init_weights()

        # Create dummy input
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            output = model(tokens)

        expected_shape = (batch_size, seq_len, model_args.vocab_size)
        if output.shape == expected_shape:
            print(f"✓ PASSED: Output shape is correct: {output.shape}")
        else:
            print(f"✗ FAILED: Expected shape {expected_shape}, got {output.shape}")
            return False

        print()
        return True
    except Exception as e:
        print(f"✗ FAILED: Forward pass raised exception: {e}")
        import traceback

        traceback.print_exc()
        print()
        return False


def test_parameter_count():
    """Test that parameter count is correct with multi-embedding."""
    print("Test 5: Parameter count verification")
    print("-" * 50)

    vocab_size = 1000
    dim = 512
    n_layers = 8
    interval = 4

    model_args = TransformerModelArgs(
        dim=dim,
        n_layers=n_layers,
        n_heads=8,
        vocab_size=vocab_size,
        multi_embedding_enabled=True,
        multi_embedding_interval=interval,
    )

    model = Transformer(model_args)

    # Count residual embedding parameters
    residual_emb_params = 0
    num_residual_embs = 0
    for layer in model.layers.values():
        if layer.residual_embedding is not None:
            num_residual_embs += 1
            residual_emb_params += sum(
                p.numel() for p in layer.residual_embedding.parameters()
            )

    expected_num = n_layers // interval  # Layers 4, 8
    expected_params = expected_num * vocab_size * dim

    print(f"  Number of residual embeddings: {num_residual_embs}")
    print(f"  Residual embedding parameters: {residual_emb_params:,}")
    print(f"  Expected: {expected_num} embeddings, {expected_params:,} parameters")

    if num_residual_embs == expected_num and residual_emb_params == expected_params:
        print("✓ PASSED: Parameter count is correct")
    else:
        print("✗ FAILED: Parameter count mismatch")
    print()
    return num_residual_embs == expected_num and residual_emb_params == expected_params


def main():
    """Run all tests."""
    print("=" * 50)
    print("Multi-Embedding Feature Tests")
    print("=" * 50)
    print()

    results = []
    results.append(("Multi-embedding disabled", test_multi_embedding_disabled()))
    results.append(("Multi-embedding enabled", test_multi_embedding_enabled()))
    results.append(("Multi-embedding interval", test_multi_embedding_interval()))
    results.append(("Forward pass", test_forward_pass()))
    results.append(("Parameter count", test_parameter_count()))

    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    print()
    total = len(results)
    passed = sum(results, key=lambda x: x[1])
    print(f"Total: {passed}/{total} tests passed")

    return all(result for _, result in results)


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
