"""
Activation hooks for extracting and modifying hidden states.
Works with both TransformerLens and raw HuggingFace models.
"""

import torch
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ActivationCache:
    """Container for cached activations."""
    activations: Dict[str, torch.Tensor]
    
    def __getitem__(self, key: str) -> torch.Tensor:
        return self.activations[key]
    
    def __setitem__(self, key: str, value: torch.Tensor):
        self.activations[key] = value
    
    def keys(self):
        return self.activations.keys()


class ActivationHooks:
    """
    Manage forward hooks for extracting and modifying activations.
    
    Compatible with HuggingFace models (Llama, Mistral, etc.)
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.cache = ActivationCache({})
        
    def _get_layer_module(self, layer_idx: int):
        """Get the module for a specific layer."""
        # Works for Llama, Mistral, and similar architectures
        if hasattr(self.model, 'model'):
            # For models wrapped in a class (e.g., LlamaForCausalLM)
            return self.model.model.layers[layer_idx]
        else:
            return self.model.layers[layer_idx]
    
    def _get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        if hasattr(self.model, 'model'):
            return len(self.model.model.layers)
        else:
            return len(self.model.layers)
    
    def register_extraction_hook(
        self, 
        layer_idx: int, 
        component: str = "residual"
    ) -> None:
        """
        Register a hook to extract activations from a layer.
        
        Args:
            layer_idx: Which layer to hook
            component: "residual" (after layer), "attn", or "mlp"
        """
        layer = self._get_layer_module(layer_idx)
        cache_key = f"layer_{layer_idx}_{component}"
        
        def hook_fn(module, input, output):
            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.cache[cache_key] = hidden_states.detach().clone()
        
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def register_steering_hook(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        coefficient: float = 1.0,
        token_positions: Optional[List[int]] = None
    ) -> None:
        """
        Register a hook to add a steering vector to activations.
        
        Args:
            layer_idx: Which layer to steer
            steering_vector: The steering vector to add
            coefficient: Scaling factor for the steering vector
            token_positions: Which token positions to steer (None = all)
        """
        layer = self._get_layer_module(layer_idx)
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            # Ensure steering vector is on correct device and dtype
            sv = steering_vector.to(hidden_states.device, hidden_states.dtype)
            
            # Apply steering
            if token_positions is None:
                # Steer all positions
                hidden_states = hidden_states + coefficient * sv
            else:
                # Steer only specified positions
                for pos in token_positions:
                    hidden_states[:, pos, :] = hidden_states[:, pos, :] + coefficient * sv
            
            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states
        
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.cache = ActivationCache({})
    
    def get_cached_activations(self) -> ActivationCache:
        """Return cached activations."""
        return self.cache
    
    @contextmanager
    def extraction_context(self, layers: List[int], component: str = "residual"):
        """
        Context manager for extracting activations.
        
        Usage:
            with hooks.extraction_context([15, 16, 17]) as cache:
                model(input_ids)
                activations = cache["layer_16_residual"]
        """
        try:
            for layer in layers:
                self.register_extraction_hook(layer, component)
            yield self.cache
        finally:
            self.clear_hooks()
    
    @contextmanager
    def steering_context(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        coefficient: float = 1.0
    ):
        """
        Context manager for steered generation.
        
        Usage:
            with hooks.steering_context(16, sv, coef=1.5):
                output = model.generate(input_ids)
        """
        try:
            self.register_steering_hook(layer_idx, steering_vector, coefficient)
            yield
        finally:
            self.clear_hooks()


class MultiVectorSteering:
    """
    Apply multiple steering vectors simultaneously.
    Supports different vectors at different layers.
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = ActivationHooks(model)
    
    def steer(
        self,
        steering_configs: List[Dict],
    ):
        """
        Apply multiple steering vectors.
        
        Args:
            steering_configs: List of dicts with keys:
                - "layer": int
                - "vector": torch.Tensor
                - "coefficient": float
        
        Usage:
            steerer = MultiVectorSteering(model)
            steerer.steer([
                {"layer": 14, "vector": formal_sv, "coefficient": 1.0},
                {"layer": 16, "vector": positive_sv, "coefficient": 0.8},
            ])
            output = model.generate(input_ids)
            steerer.clear()
        """
        for config in steering_configs:
            self.hooks.register_steering_hook(
                layer_idx=config["layer"],
                steering_vector=config["vector"],
                coefficient=config.get("coefficient", 1.0)
            )
    
    def clear(self):
        """Clear all steering hooks."""
        self.hooks.clear_hooks()
    
    @contextmanager
    def steering_context(self, steering_configs: List[Dict]):
        """Context manager for multi-vector steering."""
        try:
            self.steer(steering_configs)
            yield
        finally:
            self.clear()