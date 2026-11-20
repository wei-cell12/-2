---
annotations_creators: []
language: en
size_categories:
- n<1K
task_categories:
- image-classification
- object-detection
- image-segmentation
task_ids: []
pretty_name: SLAKE
tags:
- fiftyone
- image
- image-classification
- image-segmentation
- object-detection
- medical
dataset_summary: >




  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 642
  samples.


  ## Installation


  If you haven't already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub


  # Load the dataset

  # Note: other available arguments include 'max_samples', etc

  dataset = load_from_hub("Voxel51/SLAKE")


  # Launch the App

  session = fo.launch_app(dataset)

  ```
license: cc-by-4.0
---

# Dataset Card for SLAKE

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 642 samples.

![image/png](SLAKE.gif)

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/SLAKE")

# Launch the App
session = fo.launch_app(dataset)
```


# SLAKE Dataset Processing Overview

See the accompanying code in this repo for details on how the dataset was parsed.

## Original Structure

```
SLAKE/imgs/xmlab0/
├── detection.json       # Organ bounding boxes (absolute coordinates)
├── question.json        # VQA pairs (English + Chinese)
├── mask_xmlab0.png      # Segmentation mask (pixel values)
└── source_xmlab0.jpg    # Medical image
```

## Key Processing Steps

1. **Coordinate Conversion**: Converted absolute bounding boxes to relative coordinates for FiftyOne
2. **Language Filtering**: Extracted English-only questions while preserving order
3. **Mask Labeling**: Used mapping file to convert pixel values (70, 135, etc.) to organ names ("Liver", "Left Kidney")
4. **Data Structuring**: 
   - Sample-level: location, modality, base_type
   - Individual Q&A: question_0/answer_0, question_1/answer_1, etc.

## Final FiftyOne Dataset
Each sample contains:
- Medical image with object detections (organ bounding boxes)
- Multi-class segmentation with readable organ labels  
- VQA pairs as individual fields
- Sample metadata (imaging modality, body region, etc.)


## Dataset Details

### Dataset Description

SLAKE (Semantically-Labeled Knowledge-Enhanced) is a large-scale bilingual (English and Chinese) dataset for Medical Visual Question Answering (Med-VQA). It consists of radiology images across multiple modalities (CT, MRI, X-Ray) with detailed semantic annotations, comprehensive question-answer pairs, and a structured medical knowledge graph. SLAKE covers five body parts (head, neck, chest, abdomen, and pelvic cavity) and includes both visual reasoning and knowledge-based reasoning tasks. The dataset is specifically designed to advance the development of medical AI systems that can understand clinical images and answer medical questions.

- **Curated by:** Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, Xiao-Ming Wu from The Hong Kong Polytechnic University, West China Hospital of Sichuan University, and Sichuan Academy of Medical Sciences/Sichuan Provincial People's Hospital
- **Funded by:** Research grant P0030935 (ZVPY) funded by PolyU (UGC)
- **Shared by:** The research team from Hong Kong Polytechnic University
- **Language(s) (NLP):** English (en) 
- **License:** CC-BY-4.0

### Dataset Sources

- **Repository:** http://www.med-vqa.com/slake
- **Dataset repo on Hugging Face:** huggingface.co/datasets/bokelvin/slake
- **Paper:** [SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering](https://arxiv.org/abs/2102.09542)

## Uses

### Direct Use

SLAKE is designed for training and evaluating Medical Visual Question Answering (Med-VQA) systems. These systems can be used to:
- Assist medical professionals in diagnosis by providing a second opinion
- Help patients better understand their medical imaging results
- Support clinical education and training of medical students and professionals
- Serve as a benchmark for measuring progress in medical AI for visual understanding and reasoning
- Develop multimodal medical AI systems that can integrate visual information with structured medical knowledge

### Out-of-Scope Use

The dataset should not be used for:
- Making clinical decisions without expert oversight
- Directly providing medical advice to patients without physician review
- Replacing radiologists or other medical professionals
- Applications outside the medical domain for which the dataset was not designed
- Claiming diagnostic capabilities without proper clinical validation

## Dataset Structure

The SLAKE dataset consists of:

- **Images**: 642 radiology images (282 CT scans, 181 MRI scans, 179 X-rays) covering five body parts (head, neck, chest, abdomen, pelvic cavity)

- **Question-Answer Pairs**: 14,028 QA pairs in both English and Chinese, categorized into 10 content types:
  - Plane (1,280 questions)
  - Quality (762 questions)
  - Modality (1,492 questions)
  - Position (2,678 questions)
  - Organ (3,041 questions)
  - Knowledge Graph (KG) (1,740 questions)
  - Abnormal (1,696 questions)
  - Color (647 questions)
  - Shape (245 questions)
  - Size (447 questions)

- **Semantic Annotations**:
  - Segmentation masks for 39 organs and 12 diseases
  - Bounding box annotations for all important structures

- **Knowledge Graph**:
  - 2,603 English and 2,629 Chinese triplets in <head, relation, tail> format
  - Covers medical concepts like organ functions, disease symptoms, treatments, etc.

- **Dataset Splits**: 70% training (450 images, 9,849 questions), 15% validation (96 images, 2,109 questions), 15% testing (96 images, 2,070 questions)

## Dataset Creation

### Curation Rationale

SLAKE was created to address significant limitations in existing Med-VQA datasets:
- Lack of semantic labels (segmentation masks, bounding boxes) for training models to find regions of interest
- Absence of external knowledge resources for answering complex clinical questions
- Limited coverage of body parts and imaging modalities
- Need for multilingual support to broaden applications

The dataset aims to facilitate more sophisticated medical visual reasoning systems that can leverage both visual content and structured medical knowledge.

### Source Data

#### Data Collection and Processing

- Images were selected from three open-source datasets:
  1. Medical Segmentation Decathlon
  2. ChestXray-NIHCC (National Institutes of Health Clinical Center)
  3. CHAOS (Combined CT-MR Healthy Abdominal Organ Segmentation Challenge)

- 179 chest X-rays were randomly selected from ChestXray-NIHCC with original disease labels preserved
- 463 single-slice images were randomly chosen from 3D volume cases in other datasets
- For the knowledge graph, medical triplets were extracted from OwnThink (a Wikipedia-based knowledge base), then manually filtered and refined

#### Who are the source data producers?

- The original medical images came from public healthcare research repositories
- The knowledge graph information was derived from OwnThink/Wikipedia
- Questions and annotations were created by experienced physicians and the research team

### Annotations

#### Annotation process

- Experienced physicians labeled organs and diseases using ITK-SNAP software
- The team developed a custom annotation system with question templates for each body part
- Physicians could choose from candidate questions or amend/rewrite them based on clinical experience
- Questions were semantically labeled to distinguish between vision-only and knowledge-based types
- Answer distribution was deliberately balanced to reduce statistical bias (e.g., yes/no questions were designed to have 50-50 distribution)

#### Who are the annotators?

- Experienced physicians performed the image annotations and helped create clinically relevant questions
- The research team developed the annotation system and constructed the knowledge graph
- Medical students may have assisted in parts of the annotation process (implied in acknowledgments)

#### Personal and Sensitive Information

The paper states that "This research study was conducted retrospectively using human subject data made available in open access" and that "Ethical approval was not required as confirmed by the license attached with the open access data." The dataset appears to use de-identified medical images from public research repositories.

## Bias, Risks, and Limitations

- The dataset is limited to five body parts and three imaging modalities
- Baseline models achieved only ~73% accuracy, indicating significant room for improvement before clinical use
- The dataset may reflect biases in the original image sources or in the question generation process
- Despite efforts to balance answers, some inherent biases may remain in the data
- Performance on this dataset alone should not be used to claim clinical efficacy
- The knowledge graph, while extensive, cannot capture all relevant medical knowledge

### Recommendations

Users should recognize that Med-VQA systems trained on this dataset require clinical validation before deployment in healthcare settings. The dataset should be used primarily for research and development purposes, with careful consideration of potential biases. Evaluation metrics beyond accuracy (such as explainability and clinical relevance) should be considered when developing systems based on this dataset.

## Citation

**BibTeX:**
```
@article{liu2021slake,
  title={SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering},
  author={Liu, Bo and Zhan, Li-Ming and Xu, Li and Ma, Lin and Yang, Yan and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2102.09542},
  year={2021}
}
```

**APA:**
Liu, B., Zhan, L.-M., Xu, L., Ma, L., Yang, Y., & Wu, X.-M. (2021). SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering. arXiv preprint arXiv:2102.09542.
