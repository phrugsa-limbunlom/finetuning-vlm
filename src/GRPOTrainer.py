import logging
import os
import re

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)

from src.AI4MathDataset import AI4MathDataset
from src.GRPOConfig import GRPOConfig
from src.RewardModel import RewardModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Main trainer class for GRPO fine-tuning with Qwen2-VL and quantization support"""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.accelerator = Accelerator(
            # cpu = config.cpu,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision='fp16' if config.fp16 else 'no'
        )

        # Initialize wandb
        if self.accelerator.is_main_process:
            wandb.init(project="qwen2vl-grpo-ai4math", config=config.__dict__)

        self.setup_quantization_config()
        self.setup_model_and_processor()
        self.setup_dataset()
        self.setup_optimizer()

        # Move everything to accelerator
        (
            self.model,
            self.ref_model,
            self.reward_model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.model,
            self.ref_model,
            self.reward_model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.scheduler
        )

    def setup_quantization_config(self):
        """Setup quantization configuration"""
        self.quantization_config = None

        # Check if quantization is enabled in config
        if hasattr(self.config, 'use_quantization') and self.config.use_quantization:
            quantization_type = getattr(self.config, 'quantization_type', '4bit')

            if quantization_type == '4bit':
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=getattr(self.config, 'use_double_quant', True),
                    bnb_4bit_quant_type=getattr(self.config, 'quant_type', "nf4"),
                    bnb_4bit_compute_dtype=torch.float16 if self.config.fp16 else torch.float32,
                )
                logger.info("Using 4-bit quantization with BitsAndBytes")

            elif quantization_type == '8bit':
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=getattr(self.config, 'int8_threshold', 6.0),
                    llm_int8_has_fp16_weight=False,
                )
                logger.info("Using 8-bit quantization with BitsAndBytes")

        else:
            logger.info("No quantization enabled")

    def setup_model_and_processor(self):
        """Initialize Qwen2-VL models and processor with quantization"""
        local_model_path = "./qwen2vl-model"

        if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
            model_path = local_model_path
            logger.info(f"Loading model from local path: {model_path}")
        else:
            model_path = self.config.model_name
            logger.info(f"Loading model from HuggingFace: {model_path}")

        # Load processor (Qwen2-VL uses a different processor)
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # print(f"Padding side: {self.tokenizer.padding_side}")
        # print(f"Pad token: {self.tokenizer.pad_token}")
        # print(f"Pad token ID: {self.tokenizer.pad_token_id}")

        # Prepare model loading arguments
        model_kwargs = {
            'torch_dtype': torch.float16 if self.config.fp16 else torch.float32,
        }

        # Add quantization config if enabled
        if self.quantization_config is not None:
            model_kwargs['quantization_config'] = self.quantization_config
            # Don't use device_map="auto" with quantization in distributed setting
            if not self.accelerator.distributed_type:
                model_kwargs['device_map'] = "auto"
        else:
            model_kwargs['device_map'] = "auto" if not self.accelerator.distributed_type else None

        # Load main model (Qwen2-VL)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Prepare model for k-bit training if quantization is used
        if self.quantization_config is not None:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=getattr(self.config, 'gradient_checkpointing', True)
            )

        if hasattr(self.model, 'gradient_checkpointing_enable') and not self.quantization_config:
            self.model.gradient_checkpointing_enable()

        # Setup LoRA with Qwen2-VL specific target modules
        # Qwen2-VL uses similar attention modules but might have different naming
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # For quantized models, we might need to target additional modules
        if self.quantization_config is not None:
            # Add more target modules for better quantized performance
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Create reference model (frozen copy) with same quantization
        self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs
        )

        if self.quantization_config is not None:
            self.ref_model = prepare_model_for_kbit_training(self.ref_model, use_gradient_checkpointing=False)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Create reward model
        self.reward_model = RewardModel(self.ref_model, self.config)

        # Enable gradient checkpointing if specified and not using quantization
        if self.config.gradient_checkpointing and self.quantization_config is None:
            self.model.gradient_checkpointing_enable()

    def setup_dataset(self):
        """Load and setup datasets"""
        logger.info("Loading AI4Math dataset...")

        dataset = load_dataset(self.config.dataset_name)

        train_data = dataset['testmini']
        eval_data = dataset['testmini']

        # Create datasets
        self.train_dataset = AI4MathDataset(
            train_data, self.processor, self.tokenizer,
            max_length=self.config.max_length, split="train"
        )
        self.eval_dataset = AI4MathDataset(
            eval_data, self.processor, self.tokenizer,
            max_length=self.config.max_length, split="eval"
        )

        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.collate_fn,
        )

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """Custom collate function for batching with Qwen2-VL"""
        # Pad sequences
        input_ids = [item['input_ids'].squeeze(0) if item['input_ids'].dim() == 2 else item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'].squeeze(0) if item['attention_mask'].dim() == 2 else item['attention_mask'] for item in batch]

        # Qwen2-VL uses pixel_values for images
        pixel_values = []
        image_grid_thw = []

        for item in batch:
            if 'pixel_values' in item and item['pixel_values'] is not None:
                pixel_values.append(item['pixel_values'])
            if 'image_grid_thw' in item and item['image_grid_thw'] is not None:
                image_grid_thw.append(item['image_grid_thw'])

        # Pad input_ids and attention_masks
        max_len = max([len(ids) for ids in input_ids])

        padded_input_ids = []
        padded_attention_masks = []

        for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            pad_length = max_len - len(ids)
            # padded_ids = torch.cat([ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])

            pad_token_tensor = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
            padded_ids = torch.cat([ids, pad_token_tensor])

            # padded_mask = torch.cat([mask, torch.zeros(pad_length)])

            pad_mask_tensor = torch.zeros(pad_length, dtype=mask.dtype)
            padded_mask = torch.cat([mask, pad_mask_tensor])

            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)

        result = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
        }

        # Handle Qwen2-VL specific image inputs
        if pixel_values:
            result['pixel_values'] = torch.stack(pixel_values)

        if image_grid_thw:
            result['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)

        return result

    def setup_optimizer(self):
        """Setup optimizer and scheduler with quantization considerations"""
        # For quantized models, we might want to use different optimizer settings
        optimizer_kwargs = {
            'lr': self.config.learning_rate,
            'weight_decay': 0.01
        }

        # Adjust optimizer for quantized models
        if self.quantization_config is not None:
            # Use 8-bit Adam optimizer for better memory efficiency with quantization
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    **optimizer_kwargs
                )
                logger.info("Using 8-bit AdamW optimizer for quantized model")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to regular AdamW")
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        else:
            self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)

        total_steps = len(self.train_dataloader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

    def compute_rewards(self, input_ids, attention_mask, pixel_values, responses):
        """Compute rewards for generated responses"""
        with torch.no_grad():
            # Simple reward based on response length and mathematical keywords
            rewards = []
            for response in responses:
                reward = 0.0

                # Basic mathematical reasoning rewards
                math_keywords = ['calculate', 'solve', 'answer', 'result', 'equals', '=']
                for keyword in math_keywords:
                    if keyword in response.lower():
                        reward += 0.1

                # Length penalty/reward
                if 10 <= len(response.split()) <= 100:
                    reward += 0.2

                # Number presence (important for math problems)
                if re.search(r'\d+', response):
                    reward += 0.3

                rewards.append(reward)

            return torch.tensor(rewards, device=input_ids.device)

    def grpo_loss(self, policy_logprobs, ref_logprobs, rewards, old_logprobs):
        """Compute GRPO loss"""
        # Compute advantages
        advantages = rewards - rewards.mean()

        # Compute importance sampling ratio
        ratio = torch.exp(policy_logprobs - old_logprobs)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - self.config.cliprange, 1 + self.config.cliprange)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # KL penalty
        kl_penalty = self.config.beta * (policy_logprobs - ref_logprobs).mean()

        total_loss = policy_loss + kl_penalty

        return total_loss, policy_loss, kl_penalty

    def generate_responses(self, batch, num_responses=4):
        """Generate multiple responses for GRPO training with Qwen2-VL"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pixel_values = batch.get('pixel_values', None)
        image_grid_thw = batch.get('image_grid_thw', None)

        responses = []
        logprobs_list = []

        with torch.no_grad():
            for _ in range(num_responses):
                # Generation parameters adjusted for Qwen2-VL
                generation_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'max_new_tokens': 128,
                    'do_sample': True,
                    'temperature': 0.7,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'return_dict_in_generate': True,
                    'output_scores': True,
                    'pixel_values': pixel_values,
                    'image_grid_thw': image_grid_thw
                }

                # Add Qwen2-VL specific parameters
                # if pixel_values is not None:
                #     generation_kwargs['pixel_values'] = pixel_values
                #
                # if image_grid_thw is not None:
                #     if image_grid_thw.dim() == 3 and image_grid_thw.size(1) == 1:
                #         image_grid_thw = image_grid_thw.squeeze(1)
                #         generation_kwargs['image_grid_thw'] = image_grid_thw
                #     else:
                #          generation_kwargs['image_grid_thw'] = image_grid_thw

                # print(image_grid_thw)
                # print("Shape after squeeze:", image_grid_thw.shape)
                # print("Content:", image_grid_thw)
                # print("Type:", type(image_grid_thw))

                # print(generation_kwargs)

                outputs = self.model.generate(**generation_kwargs)

                generated_ids = outputs.sequences[:, input_ids.shape[1]:]
                responses.append(generated_ids)

                # Compute log probabilities
                logprobs = []
                for i, score in enumerate(outputs.scores):
                    probs = F.softmax(score, dim=-1)
                    selected_logprobs = torch.log(probs.gather(1, generated_ids[:, i:i + 1]))
                    logprobs.append(selected_logprobs)

                if logprobs:
                    logprobs_list.append(torch.cat(logprobs, dim=1).mean(dim=1))

        return responses, logprobs_list

    # Compute log probabilities from logits
    def compute_action_log_probs(self, model_outputs, input_ids, attention_mask=None):
        """Compute log probabilities of actions taken"""
        if not hasattr(model_outputs, 'logits') or model_outputs.logits is None:
            return torch.tensor(0.0, device=input_ids.device)

        logits = model_outputs.logits  # [batch_size, seq_len, vocab_size]
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)

        # For casual LM, align logits and tokens for next-token prediction
        if input_ids.size(1) > 1:
            # Shift logits and tokens: predict token t+1 from position t
            shift_logits = log_probs[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
            shift_tokens = input_ids[..., 1:].contiguous() # [batch, seq_len-1]

            # Gather log probs for actual tokens
            action_log_probs = torch.gather(
                shift_logits,
                dim=-1,
                index=shift_tokens.unsqueeze(-1)
            ).squeeze(-1) # [batch_size, seq_len-1]

            # Apply attention mask if provided
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                action_log_probs = action_log_probs * shift_mask
                # Sum log probs over valid position
                return action_log_probs.sum(dim=-1) # [batch_size]
            else:
                return action_log_probs.sum(dim=-1) # [batch_size]
        else:
            return torch.tensor(0.0, device=input_ids.device)


    def train_step(self, batch):
        """Single training step"""
        # Generate responses
        responses, old_logprobs = self.generate_responses(batch, self.config.group_size)

        # Decode responses for reward computation
        decoded_responses = []
        for response_batch in responses:
            decoded_batch = self.tokenizer.batch_decode(response_batch, skip_special_tokens=True)
            decoded_responses.extend(decoded_batch)

        # Compute rewards
        rewards = self.compute_rewards(
            batch['input_ids'],
            batch['attention_mask'],
            batch.get('pixel_values', None),
            decoded_responses
        )

        # Forward pass through policy model
        forward_kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch.get('labels', None)
        }

        # Add Qwen2-VL specific inputs
        if batch.get('pixel_values', None) is not None:
            forward_kwargs['pixel_values'] = batch['pixel_values']

        if batch.get('image_grid_thw', None) is not None:
            forward_kwargs['image_grid_thw'] = batch['image_grid_thw']

        outputs = self.model(**forward_kwargs)
        

        # Forward pass through reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(**forward_kwargs)

        # Compute policy and reference log probabilities
        policy_logprobs = self.compute_action_log_probs(outputs,
                                                        batch['input_ids'],
                                                        batch.get('attention_mask'))

        ref_logprobs = self.compute_action_log_probs(ref_outputs,
                                                     batch['input_ids'],
                                                     batch.get('attention_mask'))
        # Ensure having tensor output, not scalars for batch processing
        if policy_logprobs.dim() == 0:
            policy_logprobs = policy_logprobs.unsqueeze(0).expand(batch['input_ids']).size(0)
        if ref_logprobs.dim() == 0:
            ref_logprobs = ref_logprobs.unsqueeze(0).expand(batch['input_ids']).size(0)

        # Compute GRPO loss
        if len(old_logprobs) > 0:
            old_logprobs_tensor = torch.stack(old_logprobs).mean()
            grpo_loss, policy_loss, kl_penalty = self.grpo_loss(
                policy_logprobs, ref_logprobs, rewards.mean(), old_logprobs_tensor
            )
        else:
            # Fallback: use a simple policy gradient loss if no old_logprobs
            advantage = rewards - rewards.mean() # center rewards

            policy_loss = -(policy_logprobs * advantage.detach()).mean()
            kl_penalty = torch.tensor(0.0, device=batch['input_ids'].device)
            grpo_loss = policy_loss

        return grpo_loss, policy_loss, kl_penalty, rewards.mean()

    def train(self):
        """Main training loop"""
        logger.info("Starting GRPO training with Qwen2-VL...")
        if self.quantization_config is not None:
            logger.info(f"Training with quantization enabled")

        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    loss, policy_loss, kl_penalty, avg_reward = self.train_step(batch)

                    self.accelerator.backward(loss)

                    # Gradient clipping is more important with quantized models
                    if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0 and self.accelerator.is_main_process:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/policy_loss': policy_loss.item(),
                        'train/kl_penalty': kl_penalty.item(),
                        'train/avg_reward': avg_reward.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/global_step': global_step
                    }

                    # Add memory usage if quantization is enabled
                    if self.quantization_config is not None:
                        log_dict['train/gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2

                    wandb.log(log_dict)

                # Evaluation
                if global_step % self.config.eval_steps == 0:
                    self.evaluate()

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'reward': f"{avg_reward.item():.4f}"
                })

            logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / len(self.train_dataloader):.4f}")

        # Final save
        self.save_checkpoint("final")
        logger.info("Training completed!")

    def evaluate(self):
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating",
                              disable=not self.accelerator.is_local_main_process):
                forward_kwargs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch.get('labels', None)
                }

                # Add Qwen2-VL specific inputs
                if batch.get('pixel_values', None) is not None:
                    forward_kwargs['pixel_values'] = batch['pixel_values']

                if batch.get('image_grid_thw', None) is not None:
                    forward_kwargs['image_grid_thw'] = batch['image_grid_thw']

                outputs = self.model(**forward_kwargs)

                if outputs.loss is not None:
                    total_loss += outputs.loss.item() * batch['input_ids'].size(0)
                    total_samples += batch['input_ids'].size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        if self.accelerator.is_main_process:
            wandb.log({'eval/loss': avg_loss})
            logger.info(f"Evaluation loss: {avg_loss:.4f}")

        self.model.train()

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            output_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            os.makedirs(output_dir, exist_ok=True)

            # Save the model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(output_dir)
            self.processor.save_pretrained(output_dir)

            # Save quantization config if used
            if self.quantization_config is not None:
                import json
                with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
                    json.dump({
                        "quantization_type": getattr(self.config, 'quantization_type', '4bit'),
                        "use_double_quant": getattr(self.config, 'use_double_quant', True),
                        "quant_type": getattr(self.config, 'quant_type', "nf4"),
                    }, f)

            logger.info(f"Checkpoint saved to {output_dir}")