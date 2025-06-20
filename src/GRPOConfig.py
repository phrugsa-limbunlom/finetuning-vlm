from dataclasses import dataclass


@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    dataset_name: str = "AI4Math/MathVista"
    output_dir: str = "./grpo_vlm_output"

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 512

    # GRPO specific
    beta: float = 0.1  # KL penalty coefficient
    cliprange: float = 0.2  # PPO clip range
    group_size: int = 4  # Number of responses per group

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Hardware
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0 # set to 0 to prevent multi processing
    cpu = False

    # Quantization configuration
    use_quantization: bool = True
    quantization_type: str = "4bit"  # Options: "4bit", "8bit", "none"
    use_double_quant: bool = True  # Use double quantization for 4bit
    quant_type: str = "nf4"  # Quantization type: "nf4", "fp4"
    int8_threshold: float = 6.0  # Threshold for 8bit quantization

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.quantization_type not in ["4bit", "8bit", "none"]:
            raise ValueError("quantization_type must be one of: '4bit', '8bit', 'none'")

        if self.quant_type not in ["nf4", "fp4"]:
            raise ValueError("quant_type must be one of: 'nf4', 'fp4'")

        # Disable quantization if type is "none"
        if self.quantization_type == "none":
            self.use_quantization = False

        # Adjust batch size recommendations for quantization
        if self.use_quantization and self.batch_size > 8:
            print(f"Warning: Large batch size ({self.batch_size}) with quantization may cause memory issues")

        # Ensure gradient checkpointing is enabled with quantization for memory efficiency
        if self.use_quantization:
            self.gradient_checkpointing = True