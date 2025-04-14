import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from .src.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .src.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from .src.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
from .src.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline


MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}


# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]


# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback


class LoadHiDreamModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["dev", "full", "fast"],),
            }
        }

    RETURN_TYPES = ("PIPE", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("pipe", "model_type", "guidance_scale", "num_inference_steps")
    FUNCTION = "load"
    CATEGORY = "HiDream"

    def load(self, model_type):
        config = MODEL_CONFIGS[model_type]
        pretrained_model_name_or_path = config["path"]
        scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
        
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
            LLAMA_MODEL_NAME,
            use_fast=False)
        
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16).to("cuda")

        transformer = HiDreamImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16).to("cuda")

        pipe = HiDreamImagePipeline.from_pretrained(
            pretrained_model_name_or_path, 
            scheduler=scheduler,
            tokenizer_4=tokenizer_4,
            text_encoder_4=text_encoder_4,
            torch_dtype=torch.bfloat16
        ).to("cuda", torch.bfloat16)

        pipe.transformer = transformer
        
        return (pipe, model_type, config["guidance_scale"], config["num_inference_steps"])


class GenerateHiDreamImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE",),
                "model_type": ("STRING",),
                "prompt": ("STRING", {"multiline": True, "default": "A cat holding a sign that says \"Hi-Dreams.ai\"."}),
                "resolution": (["1024 × 1024 (Square)", "768 × 1360 (Portrait)", "1360 × 768 (Landscape)", 
                                "880 × 1168 (Portrait)", "1168 × 880 (Landscape)", "1248 × 832 (Landscape)", 
                                "832 × 1248 (Portrait)"],),
                "seed": ("INT", {"default": -1, "min": -1, "max": 9999999}),
                "guidance_scale": ("FLOAT",),
                "num_inference_steps": ("INT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "used_seed")
    FUNCTION = "generate"
    CATEGORY = "HiDream"

    def parse_resolution(self, res_str):
        table = {
            "1024 × 1024 (Square)": (1024, 1024),
            "768 × 1360 (Portrait)": (768, 1360),
            "1360 × 768 (Landscape)": (1360, 768),
            "880 × 1168 (Portrait)": (880, 1168),
            "1168 × 880 (Landscape)": (1168, 880),
            "1248 × 832 (Landscape)": (1248, 832),
            "832 × 1248 (Portrait)": (832, 1248)
        }
        return table.get(res_str, (1024, 1024))

    def generate(self, pipe, model_type, prompt, resolution, seed, guidance_scale, num_inference_steps):

        # Get configuration for current model
        config = MODEL_CONFIGS[model_type]
        guidance_scale = config["guidance_scale"]
        num_inference_steps = config["num_inference_steps"]

        # Handle seed
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()

        generator = torch.Generator("cuda").manual_seed(seed)

        # Parse resolution
        height, width = self.parse_resolution(resolution)

        images = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator
        ).images
        
        return (images[0], seed)


class SaveHiDreamImage:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "output.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "save"
    CATEGORY = "HiDream"

    def save(self, image, filename):
        image.save(filename)
        print(f"Image saved to {filename}")
        return (image,)

