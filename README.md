# SEA-PACE
Semi-Supervised Underwater Image Enhancement Framework

## Abstract
The scarcity of paired data severely limits the performance and generalization of learning-based underwater image enhancement (UIE) methods, especially in scenes with complex degradations. Semi-supervised learning has emerged as a promising solution by enabling the utilization of large-scale unlabeled data. However, its effectiveness is often limited by the use of static, model-agnostic metrics for pseudo-label reliability assessment.  

To address this, we propose **SEA-PACE**, a novel semi-supervised framework that integrates **model-aware uncertainty modeling** and **self-paced consistency learning (SPCL)** to fully exploit unlabeled data for UIE. Specifically, we design a **Model-Aware Reliability Estimator (MARE)** that quantifies the uncertainty of the teacher model's predictions through Gaussian Process Regression in the latent feature space. The resulting uncertainty is transformed into reliability weights via a rank-based mapping.  

In addition, the SPCL strategy employs a loss-aware schedule to dynamically prioritize high-confidence pseudo-labels, gradually incorporating more challenging samples during training. Extensive experiments on several public UIE benchmarks demonstrate that SEA-PACE consistently surpasses state-of-the-art methods in both visual quality and generalization capability.

---

## Requirements
- Python >= 3.8  
- PyTorch >= 2.4.1  
- numpy  
- opencv-python  

---

## Testing
```bash
# 1. Download the code
# 2. Place your testing images in the "test_img/input" folder
# 3. Run:
python demo.py
# 4. Results will be saved in the "test_img/sea_pace_result" folder
