[Agent0_WorkBuddy_改造_发布包.zip](https://github.com/user-attachments/files/32798641/Agent0_WorkBuddy_._.zip)
目录
- 这是什么
- 核心机制
- 本次改造做了什么
- 快速开始
- 目录结构
- 验证结果
- 重要声明
- 许可证与致谢
这是什么
本仓库是 aiming-lab/Agent0（arXiv:2511.16043）的一个改造分支，落点在 feature/workbuddy-tool-integration。
Agent0 是一个「零数据自进化」框架：它不依赖任何人工标注数据，而是从同一个基座模型初始化出两个 Agent，让它们互相「施压」、螺旋进化。本改造把其中 Executor Agent 的求解工具从代码解释器替换为 WorkBuddy 的文件工具集，验证「协同进化流程在替换工具集后能否跑通」。
核心机制
Agent0 从同一个基座模型（论文用 Qwen3-8B-Base）初始化出两个 Agent：
角色
职责
训练算法
Curriculum Agent（出题者 π_θ）
生成恰好卡在 Executor 能力边界的前沿任务
GRPO
Executor Agent（做题者 π_φ）
用工具求解任务
ADPO（模糊度动态策略优化）
协同进化的飞轮：
Curriculum 出题 → Executor 用工具求解 → Executor 变强
        ↑                                        │
        └────────── 施压：出更难的题 ←───────────┘
工具集成是这个飞轮的引擎：正是代码解释器让 Executor 突破模型权重里固化的知识上限，任务复杂度才得以持续螺旋上升。本改造把引擎从「代码解释器」换成了「WorkBuddy 工具集」。
本次改造做了什么
两件事：
1. 工具层替换：把 Executor 的 python_code / sandbox_fusion 代码解释器，替换为 WorkBuddyTool
（暴露 read_file / write_file / list_directory，接口完全遵循原项目 BaseTool 标准契约）。
2. 任务域迁移：把 Curriculum 的「数学竞赛出题」prompt 换成「文档处理任务出题」prompt。
改动清单
文件
改动
executor_train/verl_tool/servers/tools/workbuddy_tools.py
新增：WorkBuddyTool 继承 BaseTool + @register_tool，含沙箱/真实两种后端模式
executor_train/local_inference_test.py
新增：用 transformers（替代 vLLM）跑通单条多轮工具调用链路
executor_train/curriculum_doc_task_test.py
新增：手动跑「Curriculum 出题 → Executor 工具调用」多轮循环
executor_train/_test_workbuddy_tool.py
新增：工具层单元测试（沙箱模式）
executor_train/_test_real_backend.py
新增：真实文件系统后端验证（root_dir 模式）
curriculum_train/verl/utils/dataset.py
修改：新增 doc_task_format 分支（文档任务出题 prompt）
curriculum_train/question_generate/question_generate.py
修改：新增 --task_type {math,doc} 切换
curriculum_train/examples/reward_function/curriculum_reward.py
修改：新增 format_reward_doc / accuracy_reward_doc / compute_score_doc
curriculum_train/examples/format_prompt/doc_task.jinja
新增：文档任务出题格式标记
curriculum_train/examples/format_prompt/doc_format.jinja
新增：文档任务 solver 格式模板
requirements_lightweight.txt
新增：Windows/CPU 轻量化验证依赖清单
数学基线全部保留：questioner_format、math_format.jinja、format_reward / accuracy_reward 等原逻辑未删除，便于做「数学 vs 文档」对比实验。
格式对照
数学（基线）
文档处理（迁移后）
出题 prompt
expert competition-math problem setter
expert at designing document-processing tasks
任务标签
<question>...</question>
<task>...</task>
答案标签
\boxed{final_answer}
<answer>...</answer>
正确性判分
mathruler 精确判等
token 重叠（占位，生产建议 LLM-as-judge）
快速开始
环境要求
- Windows / Linux / macOS 均可（无需 vLLM / flash-attn / triton）
- Python 3.12+（3.13 亦兼容）
- 显存 ≥ 4GB（Qwen2.5-1.5B fp16 约 3GB），无 GPU 也可纯 CPU 运行
⚠️ 原项目 requirements.txt 依赖 vllm / flash-attn / triton（Linux + CUDA 专属，Windows 无法安装），完整 RL 训练还需多卡 GPU。
本分支放弃本地训练，只跑「轻量化推理 + 工具调用」可行性验证链路。
安装
# 使用隔离 venv（建议）
python -m venv ./venv && source ./venv/bin/activate   # Windows: venv\Scripts\activate

# 安装 CPU 版 torch + transformers（不要装 vllm/flash-attn）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate tqdm safetensors sentencepiece regex
# 或直接：
pip install -r requirements_lightweight.txt
下载模型（ModelScope，国内快 10 倍）
MODEL_DIR="executor_train/models/Qwen2.5-1.5B-Instruct"
mkdir -p "$MODEL_DIR"
BASE="https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors; do
  curl -sL --retry 3 -o "$MODEL_DIR/$f" "$BASE/$f"
done
实测：huggingface.co 直连不通、hf-mirror 仅 ~380KB/s、ModelScope ~3.6MB/s（1.5B 模型 ~15 分钟）。
运行
cd Agent0/executor_train

# 1) 工具层单元测试（沙箱模式，不加载模型，秒级）
python _test_workbuddy_tool.py

# 2) 真实文件系统后端验证（root_dir 模式，不加载模型，秒级）
python _test_real_backend.py

# 3) 轻量化推理验证：单条文档任务 -> 工具调用 -> 多轮总结
WB_MODEL_ID="$(pwd)/models/Qwen2.5-1.5B-Instruct" python local_inference_test.py

# 4) 任务域迁移验证：Curriculum 出题(文档) -> Executor 工具调用，N 轮循环
WB_MODEL_ID="$(pwd)/models/Qwen2.5-1.5B-Instruct" WB_ROUNDS=3 python curriculum_doc_task_test.py
目录结构
Agent0/
├── curriculum_train/              # Curriculum Agent 训练 + 数据生成
│   ├── verl/utils/dataset.py      # ★ 出题 prompt（新增 doc_task_format 分支）
│   ├── question_generate/question_generate.py  # ★ 数据生成（新增 --task_type 切换）
│   ├── examples/reward_function/curriculum_reward.py  # ★ reward（新增 doc 系列）
│   └── examples/format_prompt/    # ★ doc_task.jinja / doc_format.jinja
├── executor_train/                # Executor Agent 训练 + 工具服务
│   ├── verl_tool/servers/tools/workbuddy_tools.py  # ★ WorkBuddyTool 适配层
│   ├── local_inference_test.py    # ★ 轻量化推理验证
│   ├── curriculum_doc_task_test.py  # ★ 任务域迁移验证
│   ├── _test_workbuddy_tool.py    # ★ 单元测试
│   └── _test_real_backend.py      # ★ 真实后端验证
└── requirements_lightweight.txt   # ★ 轻量化依赖清单
（★ 为本改造新增/修改的文件）
验证结果
在 Qwen2.5-1.5B-Instruct（CPU 推理）上，工具调用链路的实测结果分三个阶段：
1. 英文 Executor prompt → 工具调用 0%：1.5B 模型对 prompt 语言与具体性极度敏感，英文引导下不发起 <tool_call>；
2. 中文 + 具体示例 → 100% 触发：换中文并给出工具格式示例后，5/5 轮成功发起工具调用；
3. 补齐「文件约定」→ 端到端闭环：把工作区真实文件清单告知 Curriculum 后，Executor 依次 list_directory → read_file ×2 → write_file，全程读写真实文件内容并给出基于真实数据的答案。
三个如实记录的 1.5B 小模型局限：数值算错（$8400 算成 $10700）、多轮后格式漂移、绝对路径被越界防护正确拦截。
结论：工具调用链路本身是通的，瓶颈在 1.5B 小模型的算术与格式遵循能力，而非工具适配层设计。
重要声明
⚠️ 本分支是在 Windows + 4GB 显存 上、用 Qwen2.5-1.5B-Instruct 做的可行性验证，
验证「协同进化流程在替换工具集后能否跑通」。并未复现论文的 8B 模型训练与 18% 提升数据。
完整 RL 训练（GRPO/ADPO）需带 GPU 的服务器。
关于「真实 WorkBuddy 后端」：WorkBuddy 的文件读写/命令执行/文档生成是 Agent 内置能力，底层即本地文件系统操作，
不存在可被任意 Python 进程通过 HTTP 调用的「WorkBuddy 文件 API」。WorkBuddyTool 已提供 root_dir 真实模式
（直接读写用户指定真实目录，越界防护生效），由 _test_real_backend.py 验证。若需云端持久化，可接 workbuddy_cloud_service。
许可证与致谢
- 本仓库基于 aiming-lab/Agent0，遵循其 Apache-2.0 许可证。
- 论文：Agent0: ...（arXiv:2511.16043，https://arxiv.org/abs/2511.16043）。
- 模型：Qwen2.5-1.5B-Instruct（Apache-2.0）。
