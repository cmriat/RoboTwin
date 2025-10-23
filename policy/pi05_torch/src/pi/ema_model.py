import torch
import torch.nn as nn
# ------------------------- EMA utilities -------------------------
class EMAModel:
    """Exponential Moving Average of model parameters for improved generalization.

    This class maintains a shadow copy of the model parameters and updates them
    using exponential moving average during training.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        """Initialize EMA model.

        Args:
            model: The model to track
            decay: EMA decay rate (default: 0.999)
        """
        self.decay = decay
        self.shadow = {}
        self.register(model)

    def register(self, model: torch.nn.Module) -> None:
        """Register model parameters for EMA tracking."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters.

        EMA formula: ema_param = decay * ema_param + (1 - decay) * model_param

        Note: For FSDP, both model params and shadow params are DTensors (sharded),
        so we update directly on the shards without gathering.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Update shadow parameter directly (works for both regular tensors and DTensors)
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        """Return EMA state dictionary for checkpointing."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Apply EMA parameters to the model (for evaluation/inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore original model parameters (after evaluation)."""
        # This would require storing original params, which we skip for now
        # since we primarily use EMA for checkpoint saving
        pass