import json
from urllib import request, parse
import random

#This is the ComfyUI api prompt format.

#If you want it for a specific workflow you can "enable dev mode options"
#in the settings of the UI (gear beside the "Queue Size: ") this will enable
#a button on the UI to save workflows in api format.

#keep in mind ComfyUI is pre alpha software so this format will change a bit.

#this is the one for the default workflow
prompt_text = """
{
  "1": {
    "inputs": {
      "ckpt_name": "pornmasterPro_proDPOV1.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "2": {
    "inputs": {
      "vae_name": "vae-ft-mse-840000-ema-pruned.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "3": {
    "inputs": {
      "ipadapter_file": "ip-adapter-full-face_sd15.bin"
    },
    "class_type": "IPAdapterModelLoader",
    "_meta": {
      "title": "Load IPAdapter Model"
    }
  },
  "4": {
    "inputs": {
      "clip_name": "clip_vision.safetensors"
    },
    "class_type": "CLIPVisionLoader",
    "_meta": {
      "title": "Load CLIP Vision"
    }
  },
  "5": {
    "inputs": {
      "weight": 0.8,
      "noise": 0.44,
      "weight_type": "original",
      "start_at": 0,
      "end_at": 1,
      "unfold_batch": false,
      "ipadapter": [
        "3",
        0
      ],
      "clip_vision": [
        "4",
        0
      ],
      "image": [
        "20",
        0
      ],
      "model": [
        "1",
        0
      ]
    },
    "class_type": "IPAdapterApply",
    "_meta": {
      "title": "Apply IPAdapter"
    }
  },
  "6": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.08.44 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "7": {
    "inputs": {
      "text": "A photo of a woman sitting on the edge of her bed, she should be fully naked, her legs should be spread and her vagina should be clearly visible. She should be masturbating with a dildo sex toy in her vagina. dildomasturbation, vaginal object insertion, female masturbation, object insertion, masturbation",
      "clip": [
        "77",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "text": "worst quality,low quality,illustration,3d,2d,painting,cartoons, sketch,condom on penis",
      "clip": [
        "77",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "9": {
    "inputs": {
      "seed": 224627116775058,
      "steps": 75,
      "cfg": 6,
      "sampler_name": "ddpm",
      "scheduler": "exponential",
      "denoise": 1,
      "model": [
        "5",
        0
      ],
      "positive": [
        "7",
        0
      ],
      "negative": [
        "8",
        0
      ],
      "latent_image": [
        "10",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "10": {
    "inputs": {
      "width": 512,
      "height": 912,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "11": {
    "inputs": {
      "samples": [
        "9",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "12": {
    "inputs": {
      "filename_prefix": "Full_Face",
      "images": [
        "11",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "20": {
    "inputs": {
      "interpolation": "LANCZOS",
      "crop_position": "top",
      "sharpening": 0.05,
      "image": [
        "73",
        0
      ]
    },
    "class_type": "PrepImageForClipVision",
    "_meta": {
      "title": "Prepare Image For Clip Vision"
    }
  },
  "21": {
    "inputs": {
      "weight": 0.99,
      "noise": 0.44,
      "weight_type": "original",
      "start_at": 0,
      "end_at": 1,
      "unfold_batch": false,
      "ipadapter": [
        "24",
        0
      ],
      "clip_vision": [
        "4",
        0
      ],
      "image": [
        "38",
        0
      ],
      "model": [
        "77",
        0
      ],
      "insightface": [
        "48",
        0
      ]
    },
    "class_type": "IPAdapterApply",
    "_meta": {
      "title": "Apply IPAdapter"
    }
  },
  "23": {
    "inputs": {
      "seed": 224627116775058,
      "steps": 100,
      "cfg": 4,
      "sampler_name": "ddpm",
      "scheduler": "exponential",
      "denoise": 1,
      "model": [
        "21",
        0
      ],
      "positive": [
        "7",
        0
      ],
      "negative": [
        "8",
        0
      ],
      "latent_image": [
        "10",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "24": {
    "inputs": {
      "ipadapter_file": "ip-adapter-faceid_sd15.bin"
    },
    "class_type": "IPAdapterModelLoader",
    "_meta": {
      "title": "Load IPAdapter Model"
    }
  },
  "26": {
    "inputs": {
      "lora_name": "ip-adapter-faceid_sd15_lora.safetensors",
      "strength_model": 0.65,
      "model": [
        "1",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "27": {
    "inputs": {
      "samples": [
        "23",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "28": {
    "inputs": {
      "filename_prefix": "FaceID",
      "images": [
        "27",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "29": {
    "inputs": {
      "weight": 0.4,
      "noise": 0.44,
      "weight_type": "original",
      "start_at": 0,
      "end_at": 1,
      "unfold_batch": false,
      "ipadapter": [
        "37",
        0
      ],
      "clip_vision": [
        "4",
        0
      ],
      "image": [
        "20",
        0
      ],
      "model": [
        "21",
        0
      ]
    },
    "class_type": "IPAdapterApply",
    "_meta": {
      "title": "Apply IPAdapter"
    }
  },
  "30": {
    "inputs": {
      "seed": 224627116775058,
      "steps": 75,
      "cfg": 6,
      "sampler_name": "ddpm",
      "scheduler": "exponential",
      "denoise": 1,
      "model": [
        "29",
        0
      ],
      "positive": [
        "7",
        0
      ],
      "negative": [
        "8",
        0
      ],
      "latent_image": [
        "10",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "31": {
    "inputs": {
      "samples": [
        "30",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "32": {
    "inputs": {
      "filename_prefix": "FaceID_Plus_Face",
      "images": [
        "31",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "33": {
    "inputs": {
      "weight": 0.8,
      "noise": 0.33,
      "weight_type": "original",
      "start_at": 0,
      "end_at": 1,
      "unfold_batch": false,
      "ipadapter": [
        "37",
        0
      ],
      "clip_vision": [
        "4",
        0
      ],
      "image": [
        "20",
        0
      ],
      "model": [
        "1",
        0
      ]
    },
    "class_type": "IPAdapterApply",
    "_meta": {
      "title": "Apply IPAdapter"
    }
  },
  "34": {
    "inputs": {
      "seed": 224627116775058,
      "steps": 75,
      "cfg": 6,
      "sampler_name": "ddpm",
      "scheduler": "exponential",
      "denoise": 1,
      "model": [
        "33",
        0
      ],
      "positive": [
        "7",
        0
      ],
      "negative": [
        "8",
        0
      ],
      "latent_image": [
        "10",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "35": {
    "inputs": {
      "samples": [
        "34",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "36": {
    "inputs": {
      "filename_prefix": "Plus_Face",
      "images": [
        "35",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "37": {
    "inputs": {
      "ipadapter_file": "ip-adapter-plus-face_sd15.bin"
    },
    "class_type": "IPAdapterModelLoader",
    "_meta": {
      "title": "Load IPAdapter Model"
    }
  },
  "38": {
    "inputs": {
      "crop_position": "top",
      "sharpening": 0,
      "pad_around": false,
      "image": [
        "73",
        0
      ]
    },
    "class_type": "PrepImageForInsightFace",
    "_meta": {
      "title": "Prepare Image For InsightFace"
    }
  },
  "41": {
    "inputs": {
      "weight": 0.4,
      "noise": 0.44,
      "weight_type": "original",
      "start_at": 0,
      "end_at": 1,
      "unfold_batch": false,
      "ipadapter": [
        "3",
        0
      ],
      "clip_vision": [
        "4",
        0
      ],
      "image": [
        "20",
        0
      ],
      "model": [
        "21",
        0
      ]
    },
    "class_type": "IPAdapterApply",
    "_meta": {
      "title": "Apply IPAdapter"
    }
  },
  "42": {
    "inputs": {
      "seed": 224627116775058,
      "steps": 75,
      "cfg": 6,
      "sampler_name": "ddpm",
      "scheduler": "exponential",
      "denoise": 1,
      "model": [
        "41",
        0
      ],
      "positive": [
        "7",
        0
      ],
      "negative": [
        "8",
        0
      ],
      "latent_image": [
        "10",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "43": {
    "inputs": {
      "samples": [
        "42",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "44": {
    "inputs": {
      "filename_prefix": "FaceID_Full_Face",
      "images": [
        "43",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "48": {
    "inputs": {
      "provider": "CUDA"
    },
    "class_type": "InsightFaceLoader",
    "_meta": {
      "title": "Load InsightFace"
    }
  },
  "49": {
    "inputs": {
      "weight": 0.9500000000000001,
      "noise": 0.44,
      "weight_type": "original",
      "start_at": 0,
      "end_at": 1,
      "unfold_batch": false,
      "ipadapter": [
        "50",
        0
      ],
      "clip_vision": [
        "4",
        0
      ],
      "image": [
        "38",
        0
      ],
      "model": [
        "75",
        0
      ],
      "insightface": [
        "48",
        0
      ]
    },
    "class_type": "IPAdapterApply",
    "_meta": {
      "title": "Apply IPAdapter"
    }
  },
  "50": {
    "inputs": {
      "ipadapter_file": "ip-adapter-faceid-plus_sd15.bin"
    },
    "class_type": "IPAdapterModelLoader",
    "_meta": {
      "title": "Load IPAdapter Model"
    }
  },
  "51": {
    "inputs": {
      "lora_name": "ip-adapter-faceid-plus_sd15_lora.safetensors",
      "strength_model": 0.65,
      "model": [
        "1",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "53": {
    "inputs": {
      "seed": 224627116775058,
      "steps": 75,
      "cfg": 7,
      "sampler_name": "ddpm",
      "scheduler": "exponential",
      "denoise": 1,
      "model": [
        "49",
        0
      ],
      "positive": [
        "7",
        0
      ],
      "negative": [
        "8",
        0
      ],
      "latent_image": [
        "10",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "54": {
    "inputs": {
      "samples": [
        "53",
        0
      ],
      "vae": [
        "2",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "55": {
    "inputs": {
      "filename_prefix": "FaceID_Plus",
      "images": [
        "54",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "56": {
    "inputs": {
      "image1": [
        "62",
        0
      ],
      "image2": [
        "64",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "57": {
    "inputs": {
      "image1": [
        "56",
        0
      ],
      "image2": [
        "61",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "58": {
    "inputs": {
      "image1": [
        "57",
        0
      ],
      "image2": [
        "63",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "59": {
    "inputs": {
      "image1": [
        "58",
        0
      ],
      "image2": [
        "60",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "60": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.09.16 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "61": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.08.55 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "62": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.09.24 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "63": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.10.21 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "64": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.09.02 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "65": {
    "inputs": {
      "image1": [
        "59",
        0
      ],
      "image2": [
        "6",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "66": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.08.21 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "67": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.08.02 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "68": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.07.54 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "69": {
    "inputs": {
      "image": "Screenshot 2024-03-01 at 2.09.09 AM.png.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "70": {
    "inputs": {
      "image1": [
        "65",
        0
      ],
      "image2": [
        "67",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "71": {
    "inputs": {
      "image1": [
        "70",
        0
      ],
      "image2": [
        "69",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "72": {
    "inputs": {
      "image1": [
        "71",
        0
      ],
      "image2": [
        "66",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "73": {
    "inputs": {
      "image1": [
        "72",
        0
      ],
      "image2": [
        "68",
        0
      ]
    },
    "class_type": "ImageBatch",
    "_meta": {
      "title": "Batch Images"
    }
  },
  "74": {
    "inputs": {
      "switch_1": "On",
      "lora_name_1": "Realistic_Visionary_NSFW-07.safetensors",
      "model_weight_1": 0.5,
      "clip_weight_1": 1,
      "switch_2": "On",
      "lora_name_2": "GodPussy1 v4.safetensors",
      "model_weight_2": 0.5,
      "clip_weight_2": 1,
      "switch_3": "On",
      "lora_name_3": "Masturbation with Dildo v1.1.safetensors",
      "model_weight_3": 0.5,
      "clip_weight_3": 1
    },
    "class_type": "CR LoRA Stack",
    "_meta": {
      "title": "💊 CR LoRA Stack"
    }
  },
  "75": {
    "inputs": {
      "model": [
        "51",
        0
      ],
      "clip": [
        "1",
        1
      ],
      "lora_stack": [
        "74",
        0
      ]
    },
    "class_type": "CR Apply LoRA Stack",
    "_meta": {
      "title": "💊 CR Apply LoRA Stack"
    }
  },
  "77": {
    "inputs": {
      "model": [
        "26",
        0
      ],
      "clip": [
        "1",
        1
      ],
      "lora_stack": [
        "74",
        0
      ]
    },
    "class_type": "CR Apply LoRA Stack",
    "_meta": {
      "title": "💊 CR Apply LoRA Stack"
    }
  }
}
"""

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


prompt = json.loads(prompt_text)
#set the text prompt for our positive CLIPTextEncode
prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

#set the seed for our KSampler node
prompt["3"]["inputs"]["seed"] = 5


queue_prompt(prompt)


