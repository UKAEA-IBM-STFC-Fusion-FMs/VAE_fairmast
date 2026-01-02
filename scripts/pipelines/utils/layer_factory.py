import torch.nn as nn
from typing import Dict, Type, Any, List


class LayerFactory:
    """
    Factory for creating PyTorch layers from specifications.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[nn.Module]] = {}
        self._register_default_layers()
    
    def _register_default_layers(self):
        """Register commonly used PyTorch layers."""
        # Convolutional layers
        self.register("conv1d", nn.Conv1d)
        self.register("conv2d", nn.Conv2d)
        self.register("conv_transpose1d", nn.ConvTranspose1d)
        self.register("conv_transpose2d", nn.ConvTranspose2d)
        
        # Linear layers
        self.register("linear", nn.Linear)
        
        # Activation functions
        self.register("relu", nn.ReLU)
        self.register("leaky_relu", nn.LeakyReLU)
        self.register("tanh", nn.Tanh)
        self.register("sigmoid", nn.Sigmoid)
        self.register("gelu", nn.GELU)
        
        # Normalization layers
        self.register("batch_norm1d", nn.BatchNorm1d)
        self.register("batch_norm2d", nn.BatchNorm2d)
        self.register("layer_norm", nn.LayerNorm)
        
        # Dropout
        self.register("dropout", nn.Dropout)
        self.register("dropout1d", nn.Dropout1d)
        self.register("dropout2d", nn.Dropout2d)
        
        # Pooling
        self.register("max_pool1d", nn.MaxPool1d)
        self.register("avg_pool1d", nn.AvgPool1d)
        self.register("adaptive_avg_pool1d", nn.AdaptiveAvgPool1d)
    
    def register(self, name: str, layer_class: Type[nn.Module]):
        """Register a new layer type."""
        if not isinstance(name, str):
            raise TypeError(f"Layer name must be a string, got {type(name).__name__}")
        if not issubclass(layer_class, nn.Module):
            raise TypeError(f"Layer class must be a subclass of nn.Module")
        
        self._registry[name] = layer_class
    
    def create_layer(self, name: str, *args, **kwargs) -> nn.Module:
        """Create a layer instance by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(f"Unknown layer '{name}'. Available layers: {available}")
        
        layer_class = self._registry[name]
        try:
            return layer_class(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create layer '{name}' with args={args}, kwargs={kwargs}. Error: {e}")
    
    def get_available_layers(self) -> List[str]:
        """Get list of all registered layer names."""
        return list(self._registry.keys())


layer_factory = LayerFactory()


class SequentialBuilder(nn.Module):
    """
    Build a sequential model from layer specifications.
    
    Example layer_specs:
    {
        "layers": [
            {
                "type": "conv1d",
                "params": {
                    "in_channels": 10,
                    "out_channels": 32,
                    "kernel_size": 3,
                    "padding": 1
                }
            },
            {
                "type": "relu"
            },
            {
                "type": "batch_norm1d",
                "params": {
                    "num_features": 32
                }
            }
        ]
    }
    """
    
    def __init__(self, layer_specs: Dict[str, List[Dict[str, Any]]], factory: LayerFactory = None):
        super().__init__()
        
        if factory is None:
            factory = layer_factory
        
        self.factory = factory
        self.layers = self._build_layers(layer_specs)
        self.sequence = nn.Sequential(*self.layers)
    
    def _build_layers(self, layer_specs: Dict[str, List[Dict[str, Any]]]) -> List[nn.Module]:
        """Build list of layers from specifications."""
        if "layers" not in layer_specs:
            raise ValueError("Layer specs must contain 'layers' key")
        
        layers = []
        for i, spec in enumerate(layer_specs["layers"]):
            if not isinstance(spec, dict):
                raise ValueError(f"Layer spec at index {i} must be a dict")
            
            if "type" not in spec:
                raise ValueError(f"Layer spec at index {i} must contain 'type' key")
            
            layer_type = spec["type"]
            params = spec.get("params", {})
            
            try:
                layer = self.factory.create_layer(layer_type, **params)
                layers.append(layer)
            except Exception as e:
                raise ValueError(f"Failed to create layer {i} (type: {layer_type}): {e}")
        
        return layers
    
    def forward(self, x):
        return self.sequence(x)


def create_sequential_model(layer_specs: Dict[str, List[Dict[str, Any]]]) -> SequentialBuilder:
    """Create a sequential model from layer specifications."""
    return SequentialBuilder(layer_specs)
