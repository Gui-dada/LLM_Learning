# 第 4 章：从头开始实现 GPT 模型来生成文本
&nbsp;
## 主要章节代码
- [01_main-chapter-code](01_main-chapter-code) 包含主要章节代码。
&nbsp;
## 奖励材料

- [02_performance-analysis](02_performance-analysis) 包含分析主章中实现的 GPT 模型性能的可选代码
- [03_kv-cache](03_kv-cache) 实现 KV 缓存以加速推理过程中的文本生成
- [07_moe](07_moe) 混合专家 (MoE) 的解释和实施
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 包含将 GPT 架构实现转换为 Llama 3.2 并从 Meta AI 加载预训练权重的分步指南（完成第 4 章后查看替代架构可能会很有趣，但您也可以在阅读第 5 章后将其保存）


&nbsp;
## 注意替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) 包含对分组查询注意力 (GQA) 的介绍，大多数现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）都使用它来替代常规多头注意力 (MHA)
- [05_mla](05_mla) 包含对多头潜在注意力 (MLA) 的介绍，由 DeepSeek V3 使用，作为常规多头注意力 (MHA) 的替代方案
- [06_swa](06_swa) 包含对滑动窗口注意（SWA）的介绍，Gemma 3 和其他人使用它
- [08_deltanet](08_deltanet) 门控 DeltaNet 作为流行的线性注意力变体的解释（用于 Qwen3-Next 和 Kimi Linear）


&nbsp;
## More

In the video below, I provide a code-along session that covers some of the chapter contents as supplementary material.

<br>
<br>

[![Link to the video](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
