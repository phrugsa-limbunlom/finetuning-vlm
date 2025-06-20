from io import BytesIO
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset


class AI4MathDataset(Dataset):
    """Dataset class for AI4Math data with proper Qwen2-VL support and image cropping"""

    def __init__(self, data, processor, tokenizer, max_length=2048, split="train",
                 image_size=(336, 336)):
        self.data = data
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.image_size = image_size  # Fixed image size for cropping

    def __len__(self):
        return len(self.data)

    def crop_center(self, image, crop_width, crop_height):
        """Crop image from center to specified dimensions"""
        img_width, img_height = image.size

        # Calculate crop box
        left = (img_width - crop_width) // 2
        top = (img_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        return image.crop((left, top, right, bottom))

    def resize_and_crop(self, image, target_size):
        """Resize and crop image to target size while maintaining aspect ratio"""
        # Convert target_size to width, height if it's a tuple
        if isinstance(target_size, (tuple, list)):
            target_width, target_height = target_size
        else:
            target_width = target_height = target_size

        # Get original dimensions
        orig_width, orig_height = image.size

        # Calculate scaling factor to ensure the smaller dimension fits
        scale_width = target_width / orig_width
        scale_height = target_height / orig_height
        scale = max(scale_width, scale_height)  # Use max to ensure we can crop

        # Resize image
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop to exact target size
        image = self.crop_center(image, target_width, target_height)

        return image

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process image with standardized cropping
        image = None
        if 'decoded_image' in item and item['decoded_image'] is not None:
            if isinstance(item['decoded_image'], str):
                # If image is a URL or path
                try:
                    if item['decoded_image'].startswith('http'):
                        response = requests.get(item['decoded_image'], timeout=10)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    else:
                        image = Image.open(item['decoded_image']).convert('RGB')

                    # Apply standardized cropping
                    if image is not None:
                        image = self.resize_and_crop(image, self.image_size)

                except Exception as e:

                    print(f"Failed to load image: {e}")

            else:
                image = item['decoded_image']
                if image is not None:
                    # Apply standardized cropping
                    image = self.resize_and_crop(image, self.image_size)

        # Process text
        question = item.get('question', '')
        answer = item.get('answer', '')

        # Create conversation format - only add <image> token if we have a valid image
        if self.split == "train":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]

        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]

        # Use proper chat template
        # try:
        #     text = self.processor.apply_chat_template(
        #         conversation,
        #         tokenize=False,
        #         add_generation_prompt=True if self.split != "train" else False
        #     )
        # except:
            # Fallback if chat template fails
        #
        # text = "\n".join([msg['content'] for msg in conversation])

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
                text=text,
                images=[image],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
        )

        # print("After processor fix:")
        # print("image_grid_thw shape:", inputs["image_grid_thw"].shape)
        # print("image_grid_thw content:", inputs["image_grid_thw"])

        processed_inputs = inputs
        # Flatten tensors and ensure proper format
        # processed_inputs = {}
        # for key in inputs:
        #     if inputs[key] is not None:
        #         if isinstance(inputs[key], torch.Tensor):
        #             processed_inputs[key] = inputs[key].squeeze(0)
        #         else:
        #             processed_inputs[key] = inputs[key]

        # Ensure required keys exist
        result = {
            'input_ids': processed_inputs.get('input_ids', torch.tensor([], dtype=torch.long)),
            'attention_mask': processed_inputs.get('attention_mask', torch.tensor([], dtype=torch.long)),
            'pixel_values': processed_inputs.get('pixel_values', None),
            'image_grid_thw': processed_inputs.get('image_grid_thw', None),
            'question': question,
            'answer': answer,
            'image': image
        }

        return result
