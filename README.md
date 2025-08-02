# SEA-PACE
Semi-Supervised Underwater Image Enhancement via Gaussian Process–Assisted Self-Paced Learning

## Abstract
The scarcity of paired data severely limits the performance and generalization of learning-based underwater image enhancement (UIE) methods. This challenge is particularly prominent in scenes with complex degradations. Semi-supervised learning has emerged as a promising solution by enabling the utilization of large-scale unlabeled data. However, its effectiveness is limited by the use of static, model-agnostic metrics for pseudo-label reliability assessment. To address this, we propose SEA-PACE, a novel semi-supervised framework that integrates model-aware uncertainty modeling and self-paced consistency learning (SPCL) to fully exploit unlabeled data for UIE. Specifically, we design a Model-Aware Reliability Estimator (MARE) that quantifies the uncertainty of the teacher model's predictions through Gaussian Process Regression in latent feature space. The resulting uncertainty is then transformed into reliability weights via a rank-based mapping. Additionally, we apply the SPCL strategy that employs a loss-aware schedule to dynamically prioritize high-confidence pseudo-labels, gradually incorporating more challenging samples during training. Extensive experiments on several public UIE benchmarks demonstrate that SEA-PACE consistently surpasses state-of-the-art methods in both visual quality and generalization capability. The source code will be made publicly available.

---

## Requirements
- Python 3.8  
- PyTorch 2.4.1 

---

## Testing
```bash
# 1. Download the code
# 2. Place your testing images in the "test_img/input" folder

# (Option 1) Only save enhanced results:
python demo.py
# Enhanced results will be saved in: "test_img/sea_pace_result"

# (Option 2) Save results and calculate PSNR/SSIM:
# Place the corresponding ground truth images in "test_img/GT"
python demo.py --test_score
# In addition to saving results, the script will:
# Compute PSNR and SSIM for each image
# Save the evaluation scores in: "sea_pace_score.txt"
