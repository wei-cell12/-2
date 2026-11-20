---
license: mit
task_categories:
- visual-question-answering
dataset_info:
  features:
  - name: Question
    sequence: string
  - name: Answer
    sequence: string
  - name: Question Type
    dtype: string
  - name: image
    dtype: binary
  - name: Year
    sequence: string
  - name: Paragraph
    dtype: string
  splits:
  - name: test
    num_bytes: 251492623
    num_examples: 1948
  download_size: 193053989
  dataset_size: 251492623
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering

🤗[Dataset](https://huggingface.co/datasets/ahmed-masry/ChartQAPro) | 🖥️[Code](https://github.com/vis-nlp/ChartQAPro) | 📄[Paper](https://arxiv.org/abs/2504.05506v2)

The abstract of the paper states that: 
> Charts are ubiquitous, as people often use them to analyze data, answer questions, and discover critical insights. However, performing complex analytical tasks with charts requires significant perceptual and cognitive effort. Chart Question Answering (CQA) systems automate this process by enabling models to interpret and reason with visual representations of data. However, existing benchmarks like ChartQA lack real-world diversity and have recently shown performance saturation with modern large vision-language models (LVLMs). To address these limitations, we introduce ChartQAPro, a new benchmark that includes 1,341 charts from 157 diverse sources, spanning various chart types, including infographics and dashboards, and featuring 1,948 questions in various types, such as multiple-choice, conversational, hypothetical, and unanswerable questions, to better reflect real-world challenges. Our evaluations with 21 models show a substantial performance drop for LVLMs on ChartQAPro; e.g., Claude Sonnet 3.5 scores 90.5% on ChartQA but only 55.81% on ChartQAPro, underscoring the complexity of chart reasoning. We complement our findings with detailed error analyses and ablation studies, identifying key challenges and opportunities for advancing LVLMs in chart understanding and reasoning. We release ChartQAPro at https://github.com/vis-nlp/ChartQAPro

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63efd75a5c2ceb16fc6e98fc/aMHb9hJMSuT_wucxkfKqN.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63efd75a5c2ceb16fc6e98fc/--J_ApLyqKcgec25M4h87.png)

# Evaluation Results

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63efd75a5c2ceb16fc6e98fc/tUFDovk074utoZ0q6W9x4.png)

# Evaluating New Model

Please follow the evaluation instructions in our **github repository**: https://github.com/vis-nlp/ChartQAPro

# Contact
If you have any questions about this work, please contact **[Ahmed Masry](https://ahmedmasryku.github.io/)** using the following email addresses: **amasry17@ku.edu.tr**,  **ahmed.elmasry24653@gmail.com**, or **masry20@yorku.ca**.

# Reference
Please cite our paper if you use our model in your research. 

```
@misc{masry2025chartqaprodiversechallengingbenchmark,
      title={ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering}, 
      author={Ahmed Masry and Mohammed Saidul Islam and Mahir Ahmed and Aayush Bajaj and Firoz Kabir and Aaryaman Kartha and Md Tahmid Rahman Laskar and Mizanur Rahman and Shadikur Rahman and Mehrad Shahmohammadi and Megh Thakkar and Md Rizwan Parvez and Enamul Hoque and Shafiq Joty},
      year={2025},
      eprint={2504.05506},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.05506}, 
}
```