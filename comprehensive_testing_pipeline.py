#!/usr/bin/env python3
"""
Comprehensive Testing Pipeline for ROBIN Watermarking Algorithm
==============================================================

This script implements a complete testing pipeline that:
1. Generates watermarked and clean images using the prompts_1000.json
2. Applies various attacks with different intensities matching results.json
3. Evaluates detection and attribution metrics
4. Saves results in the same format as the provided results.json

Author: GitHub Copilot
Date: August 18, 2025
"""

import argparse
import json
import os
import time
import copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ROBIN imports
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *


class AttackConfig:
    """Configuration for different attack types and intensities"""
    
    @staticmethod
    def get_attack_configs():
        """Returns all attack configurations matching results.json"""
        return {
            # Clean (no attack)
            "clean": {
                "type": "none",
                "intensity": "none",
                "params": {}
            },
            
            # Blur attacks
            "blur_mild": {
                "type": "blur",
                "intensity": "mild",
                "params": {"kernel_size": 3, "sigma": 0.5}
            },
            "blur_moderate": {
                "type": "blur", 
                "intensity": "moderate",
                "params": {"kernel_size": 5, "sigma": 1.0}
            },
            "blur_strong": {
                "type": "blur",
                "intensity": "strong", 
                "params": {"kernel_size": 7, "sigma": 1.5}
            },
            
            # JPEG compression attacks
            "jpeg_mild": {
                "type": "jpeg",
                "intensity": "mild",
                "params": {"quality": 80}
            },
            "jpeg_moderate": {
                "type": "jpeg",
                "intensity": "moderate", 
                "params": {"quality": 60}
            },
            "jpeg_strong": {
                "type": "jpeg",
                "intensity": "strong",
                "params": {"quality": 40}
            },
            "jpeg_extreme": {
                "type": "jpeg",
                "intensity": "extreme",
                "params": {"quality": 20}
            },
            
            # Rotation attacks
            "rotation_mild": {
                "type": "rotation",
                "intensity": "mild",
                "params": {"angle_degrees": 5}
            },
            "rotation_moderate": {
                "type": "rotation",
                "intensity": "moderate",
                "params": {"angle_degrees": 10}
            },
            "rotation_strong": {
                "type": "rotation",
                "intensity": "strong", 
                "params": {"angle_degrees": 15}
            },
            "rotation_extreme": {
                "type": "rotation",
                "intensity": "extreme",
                "params": {"angle_degrees": 30}
            },
            
            # Noise attacks (AWGN)
            "noise_mild": {
                "type": "awgn",
                "intensity": "mild",
                "params": {"noise_std": 0.02}
            },
            "noise_moderate": {
                "type": "awgn",
                "intensity": "moderate",
                "params": {"noise_std": 0.03}
            },
            "noise_strong": {
                "type": "awgn",
                "intensity": "strong",
                "params": {"noise_std": 0.05}
            },
            "noise_extreme": {
                "type": "awgn",
                "intensity": "extreme",
                "params": {"noise_std": 0.08}
            },
            
            # Scaling attacks
            "scaling_mild": {
                "type": "scaling",
                "intensity": "mild",
                "params": {"scale_factor": 0.9}
            },
            "scaling_moderate": {
                "type": "scaling",
                "intensity": "moderate",
                "params": {"scale_factor": 0.8}
            },
            "scaling_strong": {
                "type": "scaling",
                "intensity": "strong",
                "params": {"scale_factor": 0.7}
            },
            
            # Cropping attacks
            "cropping_mild": {
                "type": "cropping",
                "intensity": "mild",
                "params": {"crop_ratio": 0.9}
            },
            "cropping_moderate": {
                "type": "cropping",
                "intensity": "moderate",
                "params": {"crop_ratio": 0.8}
            },
            "cropping_strong": {
                "type": "cropping",
                "intensity": "strong",
                "params": {"crop_ratio": 0.7}
            },
            "cropping_extreme": {
                "type": "cropping",
                "intensity": "extreme",
                "params": {"crop_ratio": 0.6}
            },
            
            # Sharpening attacks
            "sharpening_mild": {
                "type": "sharpening",
                "intensity": "mild",
                "params": {"strength": 0.5}
            },
            "sharpening_moderate": {
                "type": "sharpening",
                "intensity": "moderate",
                "params": {"strength": 1.0}
            },
            "sharpening_strong": {
                "type": "sharpening",
                "intensity": "strong",
                "params": {"strength": 1.5}
            },
            
            # Combination attacks (presets)
            "combo_mild": {
                "type": "preset",
                "intensity": "mild",
                "params": {"preset": "mild"}
            },
            "combo_moderate": {
                "type": "preset",
                "intensity": "moderate",
                "params": {"preset": "moderate"}
            },
            "combo_strong": {
                "type": "preset",
                "intensity": "strong",
                "params": {"preset": "strong"}
            },
            "combo_extreme": {
                "type": "preset",
                "intensity": "extreme",
                "params": {"preset": "extreme"}
            }
        }


class ImageAttacks:
    """Implementation of various image attacks with configurable intensities"""
    
    @staticmethod
    def apply_attack(img1: Image.Image, img2: Image.Image, attack_config: Dict, seed: int = 42) -> Tuple[Image.Image, Image.Image]:
        """Apply attack to both clean and watermarked images"""
        set_random_seed(seed)
        
        attack_type = attack_config["type"]
        params = attack_config["params"]
        
        if attack_type == "none":
            return img1, img2
        elif attack_type == "blur":
            return ImageAttacks._apply_blur(img1, img2, params)
        elif attack_type == "jpeg":
            return ImageAttacks._apply_jpeg(img1, img2, params)
        elif attack_type == "rotation":
            return ImageAttacks._apply_rotation(img1, img2, params, seed)
        elif attack_type == "awgn":
            return ImageAttacks._apply_noise(img1, img2, params, seed)
        elif attack_type == "scaling":
            return ImageAttacks._apply_scaling(img1, img2, params, seed)
        elif attack_type == "cropping":
            return ImageAttacks._apply_cropping(img1, img2, params, seed)
        elif attack_type == "sharpening":
            return ImageAttacks._apply_sharpening(img1, img2, params)
        elif attack_type == "preset":
            return ImageAttacks._apply_preset(img1, img2, params, seed)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
    
    @staticmethod
    def _apply_blur(img1: Image.Image, img2: Image.Image, params: Dict) -> Tuple[Image.Image, Image.Image]:
        """Apply Gaussian blur"""
        sigma = params["sigma"]
        img1_blur = img1.filter(ImageFilter.GaussianBlur(radius=sigma))
        img2_blur = img2.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img1_blur, img2_blur
    
    @staticmethod
    def _apply_jpeg(img1: Image.Image, img2: Image.Image, params: Dict) -> Tuple[Image.Image, Image.Image]:
        """Apply JPEG compression"""
        quality = params["quality"]
        
        # Save and reload with JPEG compression
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1:
            img1.save(tmp1.name, 'JPEG', quality=quality)
            img1_compressed = Image.open(tmp1.name).convert('RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
            img2.save(tmp2.name, 'JPEG', quality=quality)
            img2_compressed = Image.open(tmp2.name).convert('RGB')
        
        # Clean up temp files
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)
        
        return img1_compressed, img2_compressed
    
    @staticmethod
    def _apply_rotation(img1: Image.Image, img2: Image.Image, params: Dict, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Apply rotation"""
        angle = params["angle_degrees"]
        set_random_seed(seed)
        
        # Use torchvision transforms for consistency
        transform = transforms.RandomRotation((angle, angle))
        img1_rot = transform(img1)
        img2_rot = transform(img2)
        
        return img1_rot, img2_rot
    
    @staticmethod
    def _apply_noise(img1: Image.Image, img2: Image.Image, params: Dict, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Apply Additive White Gaussian Noise"""
        noise_std = params["noise_std"]
        np.random.seed(seed)
        
        # Convert to numpy arrays
        img1_arr = np.array(img1).astype(np.float32)
        img2_arr = np.array(img2).astype(np.float32)
        
        # Add noise
        noise1 = np.random.normal(0, noise_std * 255, img1_arr.shape)
        noise2 = np.random.normal(0, noise_std * 255, img2_arr.shape)
        
        img1_noisy = np.clip(img1_arr + noise1, 0, 255).astype(np.uint8)
        img2_noisy = np.clip(img2_arr + noise2, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img1_noisy), Image.fromarray(img2_noisy)
    
    @staticmethod
    def _apply_scaling(img1: Image.Image, img2: Image.Image, params: Dict, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Apply scaling (resize and back)"""
        scale_factor = params["scale_factor"]
        original_size = img1.size
        
        # Scale down and back up
        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        
        img1_scaled = img1.resize(new_size, Image.LANCZOS).resize(original_size, Image.LANCZOS)
        img2_scaled = img2.resize(new_size, Image.LANCZOS).resize(original_size, Image.LANCZOS)
        
        return img1_scaled, img2_scaled
    
    @staticmethod
    def _apply_cropping(img1: Image.Image, img2: Image.Image, params: Dict, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Apply random cropping and resize back"""
        crop_ratio = params["crop_ratio"]
        set_random_seed(seed)
        
        # Use torchvision transforms
        transform = transforms.RandomResizedCrop(
            img1.size, 
            scale=(crop_ratio, crop_ratio), 
            ratio=(crop_ratio, crop_ratio)
        )
        
        img1_cropped = transform(img1)
        img2_cropped = transform(img2)
        
        return img1_cropped, img2_cropped
    
    @staticmethod
    def _apply_sharpening(img1: Image.Image, img2: Image.Image, params: Dict) -> Tuple[Image.Image, Image.Image]:
        """Apply image sharpening"""
        strength = params["strength"]
        
        enhancer1 = ImageEnhance.Sharpness(img1)
        enhancer2 = ImageEnhance.Sharpness(img2)
        
        img1_sharp = enhancer1.enhance(1 + strength)
        img2_sharp = enhancer2.enhance(1 + strength)
        
        return img1_sharp, img2_sharp
    
    @staticmethod
    def _apply_preset(img1: Image.Image, img2: Image.Image, params: Dict, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Apply preset combination attacks"""
        preset = params["preset"]
        
        if preset == "mild":
            # Mild: light blur + JPEG 80 + small rotation
            img1, img2 = ImageAttacks._apply_blur(img1, img2, {"sigma": 0.5})
            img1, img2 = ImageAttacks._apply_jpeg(img1, img2, {"quality": 80})
            img1, img2 = ImageAttacks._apply_rotation(img1, img2, {"angle_degrees": 2}, seed)
        
        elif preset == "moderate":
            # Moderate: blur + JPEG 60 + rotation + noise
            img1, img2 = ImageAttacks._apply_blur(img1, img2, {"sigma": 1.0})
            img1, img2 = ImageAttacks._apply_jpeg(img1, img2, {"quality": 60})
            img1, img2 = ImageAttacks._apply_rotation(img1, img2, {"angle_degrees": 5}, seed)
            img1, img2 = ImageAttacks._apply_noise(img1, img2, {"noise_std": 0.02}, seed)
        
        elif preset == "strong":
            # Strong: heavy attacks
            img1, img2 = ImageAttacks._apply_blur(img1, img2, {"sigma": 2.0})
            img1, img2 = ImageAttacks._apply_jpeg(img1, img2, {"quality": 40})
            img1, img2 = ImageAttacks._apply_rotation(img1, img2, {"angle_degrees": 15}, seed)
            img1, img2 = ImageAttacks._apply_cropping(img1, img2, {"crop_ratio": 0.8}, seed)
            img1, img2 = ImageAttacks._apply_noise(img1, img2, {"noise_std": 0.05}, seed)
        
        elif preset == "extreme":
            # Extreme: very heavy attacks
            img1, img2 = ImageAttacks._apply_blur(img1, img2, {"sigma": 3.0})
            img1, img2 = ImageAttacks._apply_jpeg(img1, img2, {"quality": 20})
            img1, img2 = ImageAttacks._apply_rotation(img1, img2, {"angle_degrees": 30}, seed)
            img1, img2 = ImageAttacks._apply_cropping(img1, img2, {"crop_ratio": 0.6}, seed)
            img1, img2 = ImageAttacks._apply_noise(img1, img2, {"noise_std": 0.08}, seed)
        
        return img1, img2


class WatermarkTester:
    """Main testing class for ROBIN watermarking pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_models()
        self.load_prompts()
        self.attack_configs = AttackConfig.get_attack_configs()
        
        # Results storage
        self.results = {}
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "images", "clean"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "images", "watermarked"), exist_ok=True)
        for attack_name in self.attack_configs.keys():
            os.makedirs(os.path.join(args.output_dir, "images", "attacked", attack_name), exist_ok=True)
    
    def setup_models(self):
        """Initialize the diffusion pipeline and reference models"""
        print(f"Setting up models on device: {self.device}")
        
        # Load diffusion model
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.args.model_id, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.args.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16' if 'stabilityai' in self.args.model_id else None,
        )
        self.pipe = self.pipe.to(self.device)
        
        # Load reference model for CLIP similarity
        if self.args.reference_model:
            self.ref_model, _, self.ref_clip_preprocess = open_clip.create_model_and_transforms(
                self.args.reference_model, 
                pretrained=self.args.reference_model_pretrain, 
                device=self.device
            )
            self.ref_tokenizer = open_clip.get_tokenizer(self.args.reference_model)
        else:
            self.ref_model = None
        
        # Load watermark
        if self.args.wm_path and os.path.exists(self.args.wm_path):
            print(f"Loading watermark from: {self.args.wm_path}")
            wm_data = torch.load(self.args.wm_path, map_location=self.device)
            self.opt_wm = wm_data['opt_wm'].to(self.device)
            self.opt_acond = wm_data['opt_acond'].to(torch.float16) if 'opt_acond' in wm_data else None
        else:
            print("Generating random watermark pattern")
            self.opt_wm = get_watermarking_pattern(self.pipe, self.args, self.device)
            self.opt_acond = None
        
        # Get watermarking mask
        init_latents = self.pipe.get_random_latents()
        self.watermarking_mask = get_watermarking_mask(init_latents, self.args, self.device)
        
        # Empty text embedding for detection
        self.text_embeddings = self.pipe.get_text_embedding("")
    
    def load_prompts(self):
        """Load prompts from the JSON file"""
        with open(self.args.prompts_file, 'r') as f:
            prompts_data = json.load(f)
        
        self.prompts = prompts_data['prompts']
        print(f"Loaded {len(self.prompts)} prompts")
    
    def generate_image_pair(self, prompt: str, seed: int) -> Tuple[Image.Image, Image.Image]:
        """Generate clean and watermarked image pair"""
        # Generate clean image
        set_random_seed(seed)
        init_latents_clean = self.pipe.get_random_latents()
        
        outputs_clean = self.pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps,
            height=self.args.image_length,
            width=self.args.image_length,
            latents=init_latents_clean,
        )
        clean_image = outputs_clean.images[0]
        
        # Generate watermarked image
        set_random_seed(seed)  # Use same seed for fair comparison
        init_latents_wm = self.pipe.get_random_latents()
        
        outputs_wm = self.pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps,
            height=self.args.image_length,
            width=self.args.image_length,
            latents=init_latents_wm,
            watermarking_mask=self.watermarking_mask,
            watermarking_steps=self.args.watermarking_steps,
            args=self.args,
            gt_patch=self.opt_wm,
            lguidance=self.args.guidance_scale,
            opt_acond=self.opt_acond,
        )
        watermarked_image = outputs_wm.images[0]
        
        return clean_image, watermarked_image
    
    def detect_watermark(self, image: Image.Image) -> Tuple[float, bool, int]:
        """
        Detect watermark in image and return confidence, detection result, and attributed model
        Returns: (confidence, is_watermarked, attributed_model_id)
        """
        start_time = time.time()
        
        # Transform image for processing
        img_tensor = transform_img(image).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        image_latents = self.pipe.get_image_latents(img_tensor, sample=False)
        
        # Perform DDIM inversion
        reversed_latents, latents_b, noise_b = self.pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.text_embeddings,
            guidance_scale=1.0,
            num_inference_steps=self.args.test_num_inference_steps,
            latents_b=[],
        )
        
        # Extract watermark features at injection step
        target_latents = latents_b[self.args.watermarking_steps] if len(latents_b) > self.args.watermarking_steps else reversed_latents
        
        # Evaluate watermark presence
        if 'complex' in self.args.w_measurement:
            target_latents_fft = torch.fft.fftshift(torch.fft.fft2(target_latents), dim=(-1, -2))
            target_patch = self.opt_wm
        else:
            target_latents_fft = target_latents
            target_patch = self.opt_wm
        
        # Compute watermark metric
        if 'l1' in self.args.w_measurement:
            wm_metric = torch.abs(target_latents_fft[self.watermarking_mask] - target_patch[self.watermarking_mask]).mean().item()
        else:
            wm_metric = F.mse_loss(target_latents_fft[self.watermarking_mask].real, target_patch[self.watermarking_mask].real).item()
        
        # Convert to confidence (lower metric = higher confidence)
        confidence = max(0.0, 1.0 - wm_metric)
        
        # Detection threshold (tune based on empirical results)
        detection_threshold = 0.5
        is_watermarked = confidence > detection_threshold
        
        # For attribution, assume single model (can be extended for multi-model scenario)
        attributed_model = 0 if is_watermarked else -1
        
        end_time = time.time()
        detection_time = end_time - start_time
        
        return confidence, is_watermarked, attributed_model, detection_time
    
    def run_attack_evaluation(self, clean_image: Image.Image, watermarked_image: Image.Image, 
                            attack_name: str, seed: int) -> Dict:
        """Run evaluation for a specific attack"""
        attack_config = self.attack_configs[attack_name]
        
        # Apply attack
        start_time = time.time()
        clean_attacked, wm_attacked = ImageAttacks.apply_attack(clean_image, watermarked_image, attack_config, seed)
        attack_time = time.time() - start_time
        
        # Detect watermarks in attacked images
        clean_conf, clean_detected, clean_attr, clean_det_time = self.detect_watermark(clean_attacked)
        wm_conf, wm_detected, wm_attr, wm_det_time = self.detect_watermark(wm_attacked)
        
        total_time = attack_time + clean_det_time + wm_det_time
        
        return {
            'clean_confidence': clean_conf,
            'clean_detected': clean_detected,
            'clean_attributed': clean_attr,
            'wm_confidence': wm_conf,
            'wm_detected': wm_detected,
            'wm_attributed': wm_attr,
            'detection_time': total_time,
            'clean_attacked_image': clean_attacked,
            'wm_attacked_image': wm_attacked
        }
    
    def calculate_metrics(self, results_list: List[Dict]) -> Dict:
        """Calculate F1, precision, recall, and attribution metrics"""
        # Detection metrics
        y_true_detection = []
        y_pred_detection = []
        
        # Attribution metrics  
        y_true_attribution = []
        y_pred_attribution = []
        
        confidences = []
        times = []
        
        for result in results_list:
            # Clean images should not be detected as watermarked
            y_true_detection.append(0)
            y_pred_detection.append(1 if result['clean_detected'] else 0)
            
            # Watermarked images should be detected
            y_true_detection.append(1)
            y_pred_detection.append(1 if result['wm_detected'] else 0)
            
            # Attribution (only for detected watermarked images)
            if result['wm_detected']:
                y_true_attribution.append(0)  # Correct model ID
                y_pred_attribution.append(result['wm_attributed'])
            
            confidences.extend([result['clean_confidence'], result['wm_confidence']])
            times.append(result['detection_time'])
        
        # Calculate detection metrics
        tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true_detection, y_pred_detection))
        fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true_detection, y_pred_detection))
        tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true_detection, y_pred_detection))
        fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true_detection, y_pred_detection))
        
        detection_f1 = f1_score(y_true_detection, y_pred_detection, zero_division=0)
        detection_precision = precision_score(y_true_detection, y_pred_detection, zero_division=0)
        detection_recall = recall_score(y_true_detection, y_pred_detection, zero_division=0)
        
        # Calculate attribution metrics
        if y_true_attribution:
            attr_f1_macro = f1_score(y_true_attribution, y_pred_attribution, average='macro', zero_division=0)
            attr_f1_micro = f1_score(y_true_attribution, y_pred_attribution, average='micro', zero_division=0)
            attr_precision = precision_score(y_true_attribution, y_pred_attribution, average='macro', zero_division=0)
            attr_recall = recall_score(y_true_attribution, y_pred_attribution, average='macro', zero_division=0)
            attr_accuracy = accuracy_score(y_true_attribution, y_pred_attribution)
            correct_attributions = sum(1 for yt, yp in zip(y_true_attribution, y_pred_attribution) if yt == yp)
            total_attributed = len(y_true_attribution)
        else:
            attr_f1_macro = attr_f1_micro = attr_precision = attr_recall = attr_accuracy = 0.0
            correct_attributions = total_attributed = 0
        
        return {
            'detection_metrics': {
                'f1_score': detection_f1,
                'precision': detection_precision,
                'recall': detection_recall
            },
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'attribution_metrics': {
                'f1_score_macro': attr_f1_macro,
                'f1_score_micro': attr_f1_micro,
                'precision_macro': attr_precision,
                'recall_macro': attr_recall
            },
            'attribution_accuracy': attr_accuracy,
            'correct_attributions': correct_attributions,
            'total_attributed': total_attributed,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'avg_time': np.mean(times) if times else 0.0
        }
    
    def run_comprehensive_test(self):
        """Run the complete testing pipeline"""
        print("Starting comprehensive watermark testing pipeline...")
        start_time = time.time()
        
        # Determine number of images to test
        num_images = min(self.args.num_test_images, len(self.prompts))
        
        # Sample images for balanced dataset
        watermarked_indices = list(range(0, num_images, 2))  # Even indices for watermarked
        clean_indices = list(range(1, num_images, 2))       # Odd indices for clean
        
        # Ensure equal numbers
        max_pairs = min(len(watermarked_indices), len(clean_indices))
        watermarked_indices = watermarked_indices[:max_pairs]
        clean_indices = clean_indices[:max_pairs]
        
        print(f"Testing with {max_pairs} watermarked and {max_pairs} clean images")
        
        # Generate all image pairs first
        print("Generating image pairs...")
        image_pairs = []
        
        for i, (wm_idx, clean_idx) in enumerate(tqdm(zip(watermarked_indices, clean_indices), total=max_pairs)):
            # Generate watermarked image
            wm_prompt = self.prompts[wm_idx]
            wm_seed = wm_idx + self.args.gen_seed
            _, watermarked_img = self.generate_image_pair(wm_prompt, wm_seed)
            
            # Generate clean image  
            clean_prompt = self.prompts[clean_idx]
            clean_seed = clean_idx + self.args.gen_seed
            clean_img, _ = self.generate_image_pair(clean_prompt, clean_seed)
            
            # Save base images
            clean_img.save(os.path.join(self.args.output_dir, "images", "clean", f"clean_{i:04d}.jpg"))
            watermarked_img.save(os.path.join(self.args.output_dir, "images", "watermarked", f"watermarked_{i:04d}.jpg"))
            
            image_pairs.append({
                'clean_image': clean_img,
                'watermarked_image': watermarked_img,
                'clean_prompt': clean_prompt,
                'wm_prompt': wm_prompt,
                'index': i
            })
        
        # Test each attack
        for attack_name, attack_config in tqdm(self.attack_configs.items(), desc="Testing attacks"):
            print(f"\nTesting attack: {attack_name}")
            
            attack_results = []
            
            for pair in tqdm(image_pairs, desc=f"Processing {attack_name}", leave=False):
                # Run attack evaluation
                seed = pair['index'] + self.args.gen_seed
                result = self.run_attack_evaluation(
                    pair['clean_image'], 
                    pair['watermarked_image'], 
                    attack_name, 
                    seed
                )
                attack_results.append(result)
                
                # Save attacked images
                result['clean_attacked_image'].save(
                    os.path.join(self.args.output_dir, "images", "attacked", attack_name, f"clean_{pair['index']:04d}.jpg")
                )
                result['wm_attacked_image'].save(
                    os.path.join(self.args.output_dir, "images", "attacked", attack_name, f"watermarked_{pair['index']:04d}.jpg")
                )
            
            # Calculate metrics for this attack
            metrics = self.calculate_metrics(attack_results)
            
            # Store results
            self.results[attack_name] = {
                'attack_config': attack_config,
                'sample_size': len(image_pairs) * 2,  # Total images tested
                'total_watermarked': len(image_pairs),
                'total_clean': len(image_pairs),
                **metrics,
                'intensty': attack_config['intensity']  # Note: typo matches original results.json
            }
            
            print(f"  Detection F1: {metrics['detection_metrics']['f1_score']:.3f}")
            print(f"  Attribution Accuracy: {metrics['attribution_accuracy']:.3f}")
        
        # Add benchmark info
        self.results['benchmark_info'] = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
            'total_images': len(image_pairs) * 2,
            'balanced_dataset': True,
            'watermarked_images': len(image_pairs),
            'clean_images': len(image_pairs),
            'ai_models': 1,
            'attacks_tested': len(self.attack_configs),
            'model_type': 'ROBIN Watermarking System',
            'test_scope': 'Comprehensive Pipeline + All Attacks + Balanced Dataset',
            'improvements': [
                'Automated testing pipeline',
                'Complete attack suite implementation', 
                'Balanced dataset generation',
                'Comprehensive metrics evaluation',
                'Image saving and organization'
            ]
        }
        
        # Save results
        results_path = os.path.join(self.args.output_dir, 'comprehensive_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nTesting completed in {time.time() - start_time:.2f} seconds")
        print(f"Results saved to: {results_path}")
        print(f"Images saved to: {os.path.join(self.args.output_dir, 'images')}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive ROBIN Watermark Testing Pipeline')
    
    # Core arguments
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base', 
                       help='Diffusion model ID')
    parser.add_argument('--prompts_file', default='prompts_1000.json',
                       help='Path to prompts JSON file')
    parser.add_argument('--wm_path', default=None,
                       help='Path to optimized watermark file')
    parser.add_argument('--output_dir', default='test_results',
                       help='Output directory for results and images')
    
    # Generation parameters
    parser.add_argument('--num_test_images', default=50, type=int,
                       help='Number of images to test (balanced between clean and watermarked)')
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    
    # Watermark parameters
    parser.add_argument('--watermarking_steps', default=35, type=int)
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=3, type=int)
    parser.add_argument('--w_pattern', default='ring')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='complex_l1_complex')
    parser.add_argument('--w_injection', default='complex')
    
    # Reference model for CLIP similarity
    parser.add_argument('--reference_model', default='ViT-H-14')
    parser.add_argument('--reference_model_pretrain', default='laion2b_s32b_b79k')
    
    args = parser.parse_args()
    
    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    # Run the comprehensive test
    tester = WatermarkTester(args)
    tester.run_comprehensive_test()


if __name__ == '__main__':
    main()
