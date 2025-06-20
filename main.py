import logging
import os
import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration, LlavaProcessor
)
from src.AI4MathDataset import AI4MathDataset
from src.GRPOConfig import GRPOConfig
from src.GRPOTrainer import GRPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


torch.cuda.empty_cache()

# Additional utility functions and scripts

def test_dataset_loading():
    """Test function to verify dataset loading"""
    config = GRPOConfig()
    processor = LlavaProcessor.from_pretrained(config.model_name)
    tokenizer = processor.tokenizer

    # Create dummy data
    dummy_data = [
        {
            'question': "What is 2 + 2?",
            'answer': "2 + 2 = 4",
            'image': None
        }
    ]

    dataset = AI4MathDataset(dummy_data, processor, tokenizer)
    sample = dataset[0]

    print("Dataset sample:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")


def inference_example():
    """Example inference function"""
    model_path = "./grpo_vlm_output/checkpoint-final"

    # Load model and processor
    processor = LlavaProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(model_path)

    # Example inference
    question = "What is the area of a circle with radius 5?"
    image = Image.new('RGB', (224, 224), color='white')  # Dummy image

    conversation = [
        {
            "role": "user",
            "content": f"<image>\n{question}"
        }
    ]

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Response: {response}")

def train():
    """Main function to run training"""

    # Configuration
    config = GRPOConfig(
        # model_name="llava-hf/llava-1.5-7b-hf",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        dataset_name="AI4Math/MathVista",
        output_dir="./grpo_vlm_output",
        use_quantization=True,
        quantization_type="4bit",
        use_double_quant=True,
        quant_type="nf4",
        batch_size=2,  # Adjust based on your GPU memory
        gradient_accumulation_steps=8,
        num_epochs=3,
        learning_rate=2e-5,
        gradient_checkpointing=True
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize trainer
    trainer = GRPOTrainer(config)

    # Start training
    trainer.train()

if __name__ == "__main__":
    train()