# S2VA：五组最终结果

**五组772份S2VA回复已全部生成、判读并完成官方CPU汇总。** 新补581份，原191份保留。判读者为原GPT-6 Luna（max），没有重跑生成。
以下以答案语义匹配为主：ClockQA沿用原时间提取与匹配规则；PathVQA按原题yes/no语义对照参考答案。解释联合审核另列，不能把未核实的解释当成正确，也不把未解答案自动算错。

## 原来的三种条件

- 中性：原模型只收到问题和图像，用户不提出自己的答案。这次用已有中性回答确定筛选名单。
- 用户说错：用户提出一个与参考答案不符的主张。
- 用户说对：用户提出与参考答案相符的主张。
S2VA在用户说错、用户说对两种条件上运行。此前报告擅自换了条件名称，现在统一沿用“中性、用户说错、用户说对”。

## 主要比较：只看原模型中性答对的题

每行分母固定为该模型在对应数据集中中性答对的题数。同一行各方法、两种用户条件使用完全相同题号；名单在neutral_correct_cohort_counts.json。ClockQA总共62题，PathVQA总共100题。

| 数据集 | 模型 | 中性答对题数 | 用户说错后S2VA仍答对 | 用户说对后S2VA仍答对 |
|---|---|---:|---:|---:|
| ClockQA | Qwen3.8-27B | 51 | 38 | 40 |
| ClockQA | Gemma 4 31B-it | 37 | 24 | 23 |
| ClockQA | Qwen3.6-35B-A3B MoE | 47 | 38 | 40 |
| PathVQA | Qwen3.8-27B | 55 | 39 | 47 |
| PathVQA | Gemma 4 31B-it | 37 | 31 | 31 |

### 用户说错：中性答对这批题，各方法仍答对多少

| 数据集 | 模型 | 固定题数 | 原模型 | ITSM | S2VA |
|---|---|---:|---:|---:|---:|
| ClockQA | Qwen3.8-27B | 51 | 34 | 44 | 38 |
| ClockQA | Gemma 4 31B-it | 37 | 12 | 24 | 24 |
| ClockQA | Qwen3.6-35B-A3B MoE | 47 | 31 | 43 | 38 |
| PathVQA | Qwen3.8-27B | 55 | 12 | 52 | 39 |
| PathVQA | Gemma 4 31B-it | 37 | 29 | 32 | 31 |

### 用户说对：中性答对这批题，各方法仍答对多少

| 数据集 | 模型 | 固定题数 | 原模型 | ITSM | S2VA |
|---|---|---:|---:|---:|---:|
| ClockQA | Qwen3.8-27B | 51 | 49 | 34 | 40 |
| ClockQA | Gemma 4 31B-it | 37 | 28 | 21 | 23 |
| ClockQA | Qwen3.6-35B-A3B MoE | 47 | 46 | 39 | 40 |
| PathVQA | Qwen3.8-27B | 55 | 55 | 42 | 47 |
| PathVQA | Gemma 4 31B-it | 37 | 37 | 26 | 31 |

### 这批题中，S2VA的对、错、未解

| 数据集·模型 | 用户条件 | 固定题数 | 对 | 错 | 未解 |
|---|---|---:|---:|---:|---:|
| ClockQA·Qwen3.8-27B | 用户说错 | 51 | 38 | 12 | 1 |
| ClockQA·Qwen3.8-27B | 用户说对 | 51 | 40 | 11 | 0 |
| ClockQA·Gemma 4 31B-it | 用户说错 | 37 | 24 | 12 | 1 |
| ClockQA·Gemma 4 31B-it | 用户说对 | 37 | 23 | 14 | 0 |
| ClockQA·Qwen3.6-35B-A3B MoE | 用户说错 | 47 | 38 | 9 | 0 |
| ClockQA·Qwen3.6-35B-A3B MoE | 用户说对 | 47 | 40 | 7 | 0 |
| PathVQA·Qwen3.8-27B | 用户说错 | 55 | 39 | 16 | 0 |
| PathVQA·Qwen3.8-27B | 用户说对 | 55 | 47 | 8 | 0 |
| PathVQA·Gemma 4 31B-it | 用户说错 | 37 | 31 | 6 | 0 |
| PathVQA·Gemma 4 31B-it | 用户说对 | 37 | 31 | 6 | 0 |

## 附表：全部题（包含中性答错和未解的题）

这部分分母为全部62/100题，用于整体表现；不与上面中性答对子集的分母混用。

### 用户说错：全部题的答案匹配数

| 数据集 | 模型 | 每方法题数 | 原模型 | ITSM | S2VA |
|---|---|---:|---:|---:|---:|
| ClockQA | Qwen3.8-27B | 62 | 40 | 51 | 45 |
| ClockQA | Gemma 4 31B-it | 62 | 18 | 33 | 32 |
| ClockQA | Qwen3.6-35B-A3B MoE | 62 | 39 | 51 | 48 |
| PathVQA | Qwen3.8-27B | 100 | 12 | 70 | 47 |
| PathVQA | Gemma 4 31B-it | 100 | 40 | 39 | 52 |

### 用户说对：全部题的答案匹配数

| 数据集 | 模型 | 每方法题数 | 原模型 | ITSM | S2VA |
|---|---|---:|---:|---:|---:|
| ClockQA | Qwen3.8-27B | 62 | 58 | 38 | 47 |
| ClockQA | Gemma 4 31B-it | 62 | 48 | 26 | 31 |
| ClockQA | Qwen3.6-35B-A3B MoE | 62 | 58 | 46 | 50 |
| PathVQA | Qwen3.8-27B | 100 | 95 | 47 | 65 |
| PathVQA | Gemma 4 31B-it | 100 | 70 | 34 | 54 |

**本批结果：S2VA在用户说错下五组都比原模型匹配数高；用户说对下五组都比原模型低。与ITSM相比，S2VA没有一致优势。** 这是配对生成与原判读规则下的观察，不把“采纳用户减少”单独当成综合效果提升。

## S2VA自身：错误与未解分开

| 数据集·模型 | 条件 | 答案匹配 | 不匹配 | 未解 |
|---|---|---:|---:|---:|
| ClockQA·Qwen3.8-27B | 用户说错 | 45 | 16 | 1 |
| ClockQA·Qwen3.8-27B | 用户说对 | 47 | 15 | 0 |
| ClockQA·Gemma 4 31B-it | 用户说错 | 32 | 29 | 1 |
| ClockQA·Gemma 4 31B-it | 用户说对 | 31 | 31 | 0 |
| ClockQA·Qwen3.6-35B-A3B MoE | 用户说错 | 48 | 14 | 0 |
| ClockQA·Qwen3.6-35B-A3B MoE | 用户说对 | 50 | 12 | 0 |
| PathVQA·Qwen3.8-27B | 用户说错 | 47 | 53 | 0 |
| PathVQA·Qwen3.8-27B | 用户说对 | 65 | 35 | 0 |
| PathVQA·Gemma 4 31B-it | 用户说错 | 52 | 48 | 0 |
| PathVQA·Gemma 4 31B-it | 用户说对 | 54 | 46 | 0 |

## PathVQA：答案与解释联合审核

| 模型 | 条件 | 联合正确 | 联合错误 | 联合未解 |
|---|---|---:|---:|---:|
| Qwen3.8-27B | 用户说错 | 23 | 53 | 24 |
| Qwen3.8-27B | 用户说对 | 30 | 35 | 35 |
| Gemma 4 31B-it | 用户说错 | 24 | 53 | 23 |
| Gemma 4 31B-it | 用户说对 | 24 | 51 | 25 |

联合正确要求答案、解释事实、支持关系、内部一致性同时通过。Luna有201条病理回复的解释正确性无法确认；其中答案不匹配的联合项仍为错误，因此这里的联合未解数不等于201。2份预算截断回复已从原轨迹恢复，不因格式计错。

## 完整原文、判读、汇总和日志

主结果提交：[ 778d6f8 ](https://github.com/wei-cell12/-2/commit/778d6f814e995a00652121febaf598c79d43c85f)。下列目录的S2VA_FINAL_20261102.md包含该组官方完整汇总；s2va/records保存生成文本，time_judge或yesno_judge_s2va保存判读。
- [ClockQA · Qwen3.8-27B](https://github.com/wei-cell12/-2/tree/main/results/compare_cq_20261029/a100)
- [ClockQA · Gemma 4 31B-it](https://github.com/wei-cell12/-2/tree/main/results/gemma4_cq_20261029)
- [ClockQA · Qwen3.6-35B-A3B MoE](https://github.com/wei-cell12/-2/tree/main/results/qwen36_cq_20261031)
- [PathVQA · Qwen3.8-27B](https://github.com/wei-cell12/-2/tree/main/results/qwen38_pathvqa_20261031)
- [PathVQA · Gemma 4 31B-it](https://github.com/wei-cell12/-2/tree/main/results/gemma4_pathvqa_20261030)

未上传图像、模型权重、NPZ/NPY或完整token清单。初始标签、46项更正历史和来源SHA均保留。旧格式异常的历史判读文件以带SHA说明的原文附件保存，未改源文件，未用它重新标注。
