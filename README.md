# ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization

<img src=imgs/teaser_ROBIN.png  width="90%" height="60%">

This code is the pytorch implementation of [ROBIN Watermarks](https://arxiv.org/abs/2411.03862).

If you have any questions, feel free to email <hyhuang@whu.edu.cn>.

## Abstract
Watermarking generative content serves as a vital tool for authentication, ownership protection, and mitigation of potential misuse. Existing watermarking methods face the challenge of balancing robustness and concealment. They empirically inject a watermark that is both invisible and robust and *passively* achieve concealment by limiting the strength of the watermark, thus reducing the robustness. In this paper, we propose to explicitly introduce a watermark hiding process to *actively* achieve concealment, thus allowing the embedding of stronger watermarks. To be specific, we implant a robust watermark in an intermediate diffusion state and then guide the model to hide the watermark in the final generated image. We employ an adversarial optimization algorithm to produce the optimal hiding prompt guiding signal for each watermark. The prompt embedding is optimized to minimize artifacts in the generated image, while the watermark is optimized to achieve maximum strength. The watermark can be verified by reversing the generation process. Experiments on various diffusion models demonstrate the watermark remains verifiable even under significant image tampering and shows superior invisibility compared to other state-of-the-art robust watermarking methods.



## Dependencies
- PyTorch == 1.13.0
- transformers == 4.23.1
- diffusers == 0.11.1
- datasets

Note: higher diffusers version may not be compatible with the DDIM inversion code.

## Usage

### Perform adversarial optimization
```
python gen_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --model_id $path_to_diffusion_model
```

### Perform watermark embedding and evaluate robustness
```
python inject_wm_inner_latent_robin.py --run_name all_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --wm_path $path_of_optimized_wm --reference_model ViT-H-14 --reference_model_pretrain $path_to_clip_model --model_id $path_to_diffusion_model
```

## Cite
Welcome to cite our work if you find it is helpful to your research.
```
@inproceedings{huangrobin,
  title={ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization},
  author={Huang, Huayang and Wu, Yu and Wang, Qian},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}

```

## Credits
- This project is inspired by [Tree-Ring Watermarks](https://github.com/YuxinWenRick/tree-ring-watermark)