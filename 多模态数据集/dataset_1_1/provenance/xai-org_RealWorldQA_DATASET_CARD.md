---
license: cc-by-nd-4.0
dataset_info:
  features:
  - name: image
    dtype: image
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_bytes: 678377246.0
    num_examples: 765
  download_size: 1356662350
  dataset_size: 1356754492.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

## RealWorldQA

RealWorldQA is a benchmark designed for real-world understanding. The dataset consists of anonymized images taken from vehicles, in addition to other real-world images. We are excited to release RealWorldQA to the community, and we intend to expand it as our multimodal models improve.

The initial release of the RealWorldQA consists of over 700 images, with a question and easily verifiable answer for each image. See the announcement of [Grok-1.5 Vision Preview](https://x.ai/blog/grok-1.5v).

### License

RealWorldQA is released under CC BY-ND 4.0.
