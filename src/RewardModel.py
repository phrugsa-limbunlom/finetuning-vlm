from torch import nn


class RewardModel(nn.Module):
    """Reward model for mathematical reasoning evaluation"""

    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model

        # Get hidden size from the language model part of LLaVA
        if hasattr(base_model.config, 'text_config'):
            hidden_size = base_model.config.text_config.hidden_size
        elif hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        elif hasattr(base_model, 'language_model'):
            hidden_size = base_model.language_model.config.hidden_size
        else:
            # Fallback for LLaVA models
            hidden_size = 4096  # Default for LLaVA-7B

        self.reward_head = nn.Linear(hidden_size, 1)
        self.config = config
        self.hidden_size = hidden_size
        self.config = config

    def forward(self, input_ids, attention_mask, pixel_values=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )

        # Use the last hidden state of the last token
        last_hidden_state = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        reward_values = self.reward_head(
            last_hidden_state[range(len(sequence_lengths)), sequence_lengths]
        )

        return reward_values.squeeze(-1)