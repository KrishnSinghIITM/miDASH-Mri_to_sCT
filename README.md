# miDASH: MRI to Synthetic CT Generation

**Medical Image Diagnostic Analysis & Synthesis Hub**

**Author:** Krishn Singh (IITMCS_2406170)
**Course:** Minor in Data Science & Machine Learning
**Institution:** Indian Institute of Technology, Mandi

---

### Overview

Computed Tomography (CT) scans provide essential structural information for diagnosis and treatment planning but expose patients to ionizing radiation. Magnetic Resonance Imaging (MRI) offers safer imaging but lacks CT’s quantitative density mapping.

That’s why we developed **miDASH** — a deep learning framework for generating **Synthetic CT (sCT)** images directly from MRI scans. It integrates a custom U-Net model and an interactive web interface for real-time MRI-to-CT translation, visualization, and analysis.

---

### Problem Statement

Design a deep learning model that can synthesize CT-equivalent images from MRI scans, preserving diagnostic fidelity while removing the need for radiation-based imaging.

---

### Objectives

* Build a U-Net–based deep learning model for MRI → CT generation.
* Evaluate similarity using **SSIM** and **PSNR** metrics.
* Reduce dependency on CT scans in radiation-sensitive workflows.
* Deploy a Flask-based clinical dashboard for visualization and inference.

---

### Dataset

* **Source:** SynthRad 2023 – Paired 3D MRI and CT volumes (NIfTI format).
* **Dataset size:** 181 patients (~100 GB)

  * Training: 144 · Validation: 18 · Testing: 19
* **Processing:**

  * Converted 3D volumes into aligned 2D MRI–CT slices.
  * Applied normalization and uniform spatial scaling.

---

### Methodology

#### Model Architecture

A custom **U-Net** was implemented in PyTorch for 2D MRI–CT image synthesis. The network features encoder–decoder blocks with skip connections for structural preservation.

![U-Net Architecture](https://github.com/KrishnSinghIITM/miDASH-Mri_to_sCT/blob/main/assets/unet_arch)

#### Training Configuration

* **Framework:** PyTorch
* **Hardware:** Google Colab GPU (NVIDIA Tesla T4, 16 GB VRAM)
* **Loss:** L1 (Mean Absolute Error)
* **Optimizer:** Adam (lr = 1e-4)
* **Batch Size:** 8
* **Epochs:** 50 (with early stopping)

#### Evaluation Metrics

* **SSIM (Structural Similarity Index):** Structural and perceptual fidelity
* **PSNR (Peak Signal-to-Noise Ratio):** Reconstruction accuracy (in dB)
* **MAE:** Pixel-wise intensity deviation

---

### Implementation

* Processed and prepared >100 GB of MRI–CT paired data.
* Trained and validated the U-Net model on Google Colab GPU.
* Integrated training, validation, and visualization pipelines.
* Deployed real-time inference via the **miDASH** dashboard.

---

### Results

| Metric        | Validation |  Test  |
| :------------ | :--------: | :----: |
| **SSIM**      |   0.9553   | 0.9515 |
| **PSNR (dB)** |    32.33   |  32.10 |
| **L1 Loss**   |  < 0.0124  |    —   |

#### Observations

* Synthetic CTs demonstrated excellent alignment with real CT scans.
* Achieved competitive SSIM and PSNR scores vs. existing research.
* GPU acceleration improved training stability and convergence.

---

**Validation Results:**

![Validation Results](https://github.com/KrishnSinghIITM/miDASH-Mri_to_sCT/blob/main/assets/result_val)

---

**Test Results:**

![Test Results](https://github.com/KrishnSinghIITM/miDASH-Mri_to_sCT/blob/main/assets/result_test)

---

### miDASH Web Application

A **Flask-based** web app for real-time MRI → sCT synthesis and medical image visualization.

**Key Features:**

* MRI → Synthetic CT conversion using trained U-Net.
* Tissue segmentation (White Matter, Gray Matter, CSF).
* Density classification (Bone, Soft Tissue, Fat, Air).
* Real-time visualization and statistical analysis.

Demo Video: [View on LinkedIn](https://www.linkedin.com/posts/krishnsinghiitm_ai-healthcareai-medicalimaging-activity-7377695890062376960-fGJY?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE_CsbQBr-0AJo16HVG__QxxowCiBBwuTsc)

![miDASH Dashboard Interface](https://github.com/KrishnSinghIITM/miDASH-Mri_to_sCT/blob/main/assets/demo_img)

---

### Conclusion

The **miDASH** system successfully demonstrates deep learning–based synthetic CT generation from MRI. With **SSIM > 0.95** and **PSNR > 32 dB**, the model achieves research-grade performance and real-time usability through an integrated web interface.

---

### Future Work

* Implement advanced generative models (MedGAN, Pix2Pix, Diffusion).
* Extend to multi-sequence MRI modalities (T1, T2, FLAIR).
* Explore 3D volumetric and transformer-based architectures.
* Integrate GPU/cloud-based real-time inference.
* Validate across multi-institutional clinical datasets.

---

### Technology Stack

| Category                 | Tools / Frameworks                   |
| ------------------------ | ------------------------------------ |
| **Language**             | Python 3.10                          |
| **Deep Learning**        | PyTorch                              |
| **Training Environment** | Google Colab GPU (Tesla T4)          |
| **Data Processing**      | NumPy, OpenCV, Scikit-image          |
| **Visualization**        | Matplotlib, Plotly, Dash             |
| **Backend**              | Flask                                |
| **Frontend / UI**        | HTML5, CSS3, JavaScript, Plotly Dash |
| **Evaluation**           | SSIM, PSNR, MAE                      |
| **Deployment**           | Flask (local) / Docker-ready         |
| **Version Control**      | Git, GitHub                          |
| **Documentation**        | Markdown, Jupyter Notebooks          |

---

### Repository Structure

```
Mri_to_sCT_Krishn_Singh_2406170_iitmd_masai_capstone_project/
│
├── miDASH_webapp/
│   └── capstone/
│       ├── static/
│       ├── template/
│       ├── uploads/
│       ├── app.py
│       ├── README.md
│       └── requirements.txt
│
├── processed_data/
│   ├── brain/
│   ├── brain_unet_training/
│   │   └── checkpoints/
│   ├── pelvis/
│   └── preped_Task1/
│
├── scripts/
│   ├── unet.ipynb
│   ├── viz.ipynb
│   └── webapp.py
│
└── report.pdf
```

---

### Contact ;-

**Krishn Singh**

Email: [o24105@cce.iitmandi.ac.in](mailto:o24105@cce.iitmandi.ac.in)

GitHub: [github.com/KrishnSinghIITM](https://github.com/KrishnSinghIITM)

LinkedIn: [linkedin.com/in/krishnsinghiitm](https://linkedin.com/in/krishnsinghiitm)

Demo Video: [View on LinkedIn](https://www.linkedin.com/posts/krishnsinghiitm_ai-healthcareai-medicalimaging-activity-7377695890062376960-fGJY?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE_CsbQBr-0AJo16HVG__QxxowCiBBwuTsc)

