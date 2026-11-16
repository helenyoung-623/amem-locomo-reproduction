# A-Mem 论文复现 · 阶段成果

截至 2026 年 11 月 16 日，已完成 A-Mem 核心流程的独立实现，并用 Ollama 的 Qwen2.5-3B 在 LoCoMo 的前 3 段对话上开展评测，共覆盖 **497 道不同问题**。第三段对话另做了检索扩展对比和单跳题引用提示对照实验。完整的 10 段对话、多模型和论文基线对照仍待复现。

论文为 Xu 等人在 NeurIPS 2025 发表的《A-Mem: Agentic Memory for LLM Agents》。[论文](https://arxiv.org/abs/2502.12110) · [作者评测仓库](https://github.com/WujiangXu/AgenticMemory) · [LoCoMo 数据集](https://github.com/snap-research/locomo)

## 成果一览

| 交付 | 内容 |
|---|---|
| [阶段报告](REPORT.md) | 方法、逐实验对比、错误案例与复现范围 |
| [核心实现](my_amem.py) | 笔记构建、向量索引、链接生成和记忆演化 |
| [评测入口](eval_my_amem.py) | LoCoMo 读取、问题检索、问答、评分和断点恢复 |
| [实验指标](results/metrics.json) | 可复核的汇总指标及逐题分数，按历史运行区分 |
| [作者辅助代码](vendor/agentic_memory) | 用于加载 LoCoMo 和计算指标的两个辅助模块，附原 MIT 许可 |

第三段对话扩展版完整命中标注证据的题数从 **71 增至 82**，平均词级 F1 从 **30.24 降至 28.35**。进一步固定笔记和温度，对 86 道单跳题重跑后，强制先引用原文的平均 F1 从 **28.59 降至 20.32**，宽松判对数持平。详情见阶段报告。

## 运行方法

准备 Python 3.11、运行中的 Ollama 服务，并从 [作者评测仓库](https://github.com/WujiangXu/AgenticMemory/tree/main/data) 取得 `locomo10.json`，放到本仓库根目录。首次运行会加载 Hugging Face 上的向量模型和 BERTScore 模型。

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab wordnet
ollama pull qwen2.5:3b
python eval_my_amem.py --author-dir vendor/agentic_memory --dataset locomo10.json --sample-start 1 --samples 1 --max-turns 5 --max-questions 1 --output runs/smoke.json
```

上面的命令用于验证环境和执行路径。正式评测时去掉两个 `--max-*` 参数并指定新的输出文件。依赖版本按当前 Windows 验证环境记录。若在 Linux GPU 平台使用预装 PyTorch，应选择与该平台驱动匹配的 PyTorch 构建。相同输出路径会加载对应检查点，配置或源代码发生变更时应换一个输出文件名。Ollama 使用本地地址，实验无需 OpenAI API Key。

```bash
python eval_my_amem.py --author-dir vendor/agentic_memory --dataset locomo10.json --sample-start 3 --samples 1 --evolution-workers 2 --linked-note-limit 10 --output runs/sample3-interleaved.json
```

这条命令执行当前实现，其中链接邻居紧跟各自的初始命中笔记。历史成绩对应报告中标注的旧版配置。仓库里的结果只包含评分和证据匹配统计，运行时生成的对话内容、原始笔记、模型权重和检查点需保存在自己的实验环境。
