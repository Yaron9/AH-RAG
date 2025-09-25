### **项目文档：Agentic Hierarchical RAG (AH-RAG)**

**1. 项目愿景与目标**

* **愿景 (Vision)**: 构建一个能够模拟人类专家研究行为的下一代RAG框架。该框架通过结合动态推理智能体与结构化多层知识库，实现从"信息检索"到"知识探索"的范式转变。
* **目标 (Goal)**: 在处理复杂、多跳、跨领域的问题时，新框架的**答案准确性**、**推理连贯性**和**检索效率**（显著降低信息冗余）将全面超越现有的RAG模型。

**2. 核心架构**

AH-RAG框架由以下四个核心模块组成：

1.  **知识环境 (Knowledge Environment)**: 一个多层级的混合知识图谱。
    * [cite_start]**L0 层**: 精细的知识超图 (Knowledge Hypergraph)，用于捕捉原始文本中复杂的N元关系 [cite: 10, 120]。
    * [cite_start]**L1+ 层**: 语义摘要网络，由L0层实体通过语义聚合算法逐层构建，并生成摘要节点间的关系，形成一个从具体到抽象、完全可导航的网络 [cite: 867, 989, 1050]。
2.  [cite_start]**AH-RAG 智能体 (Agent)**: 一个基于LLM的策略网络，通过强化学习 (RL) 进行训练，使其学会在分层知识环境中进行最优路径探索 [cite: 9]。
3.  [cite_start]**推理引擎 (Reasoning Engine)**: 驱动智能体与知识环境进行多轮交互的循环机制。在每一轮，智能体执行"思考-行动"循环，选择最优动作（如语义锚定、LCA扩展、层级遍历等）来收集证据 [cite: 10, 169, 1061]。
4.  [cite_start]**优化模块 (Optimization Module)**: 基于GRAPH-R1的端到端强化学习训练流程，采用GRPO算法 [cite: 95, 256] [cite_start]和结果导向的奖励函数 [cite: 265] 来优化智能体的决策策略。

**3. 技术栈 (Technical Stack)**

* **编程语言**: Python 3.10+
* **LLM 交互**: 统一LLM客户端管理器 (`ah_rag.utils.llm_client`)，支持多provider配置 (Kimi/DeepSeek/OpenAI/Qwen)，Hugging Face (`transformers`, `datasets`)
* **知识图谱与数据结构**: NetworkX (用于灵活的图操作), PyG (PyTorch Geometric) (若需GNN模型)
* **向量检索与数据库**: FAISS (本地高效检索), ChromaDB / Weaviate (可扩展的向量数据库)
* [cite_start]**文本嵌入模型**: Hugging Face `sentence-transformers` (如 BGE-M3 [cite: 1120] 或其他SOTA模型)
* [cite_start]**机器学习与强化学习**: PyTorch (核心训练框架), Scikit-learn (用于GMM聚类 [cite: 1037]), Hugging Face `trl` (用于RLHF/RLAIF训练，可借鉴其PPO/GRPO实现)
* **实验跟踪与版本控制**: Weights & Biases / MLflow (用于记录实验结果与模型), Git / DVC (用于代码和数据版本控制)

---

### **任务清单与实施方案 (Task Checklist & Implementation Plan)**

以下是项目的详细任务分解，按四个主要阶段进行。

#### **阶段一：知识环境构建 (Milestone 1: Knowledge Environment Construction)**

*目标：构建一个稳定、高效、可导航的分层知识超图。*

- **[x] 任务 1.1: L0 知识超图提取器**
    * **实现方案**: 采用统一LLM客户端管理器实现高效、可配置的结构化知识提取。在`hypergraph_schema.py`中用Pydantic定义`HypergraphExtraction`和`Entity`的数据模型。创建`HypergraphExtractor`类，使用精心设计的Prompt模板，将长文档自动分块并提取N元关系，输出为包含超边、关系、实体和置信度分数的结构化JSON。
    * **技术栈**: Python, `ah_rag.utils.llm_client`, `pydantic`.

- **[x] 任务 1.2: 分层语义聚合器 (V1)**
    * **实现方案**: 采用三阶段方法构建L1摘要层。**(1) 实体嵌入与软聚类**: 使用`sentence-transformers`为所有L0实体生成向量嵌入，采用`BERTopic`对这些嵌入进行主题建模（软聚类），允许每个L0实体根据主题相关性概率分配到多个L1父节点，形成DAG结构。**(2) 摘要生成**: 对BERTopic识别出的每个主题（簇），收集其成员实体的所有描述文本，然后通过统一LLM客户端调用来生成该簇的摘要节点标题与描述。**(3) 关系生成与验证**: 使用基于L0节点重叠度的启发式规则来生成L1摘要节点间的关系。引入`LLM-as-a-Judge`机制，随机抽样并要求强LLM对生成的节点和关系进行打分，以验证和保证该层的知识质量。
    * **技术栈**: `sentence-transformers`, `bertopic`, `umap-learn`, `hdbscan`, `ah_rag.utils.llm_client`, `numpy`.
    * **产物**: `artifacts/embeddings.npy`, `artifacts/topics.json`, `artifacts/l1_nodes.json`, `artifacts/l1_summaries.json`, `artifacts/l1_edges.json`, `artifacts/l1_judge_nodes.json`, `artifacts/l1_judge_edges.json`, `artifacts/l2_nodes.json`, `artifacts/l1_to_l2.json`。

- **[x] 任务 1.3: 统一图数据结构 (NetworkX)**
    * **实现方案**: 用单一 `networkx.DiGraph` 统一承载 L0 实体(`entity`)、L0 超边(`hyperedge`)与 L1+ 摘要(`summary`)，并以边属性表达参与(`participates_in`)、层级归属(`belongs_to`)与摘要间关系(`related_to`)。节点/边携带完整元数据（置信度、诊断、评审分等），结构以 GraphML/JSON-Graph 持久化，大向量单独存储（npy/memmap）。
    * **技术栈**: `networkx`, `numpy`, `json`。

- **[x] 任务 1.4: 节点嵌入与向量索引（优化版）**
    * **实现方案**: 集成化多层向量索引与混合检索。为图中 `entity`(L0) 与 `summary`(L1) 节点统一生成与索引嵌入；利用元数据区分层与类型，并在检索后结合图结构做扩展与重排序，实现"宏观概念→微观事实"的多层检索与语义锚定。
    * **技术栈**: `sentence-transformers`, `ChromaDB`, `numpy`。

#### **阶段二：智能体与动作空间实现 (Milestone 2: Agent & Action Space Implementation)**

*目标：创建一个能够与分层图环境交互的智能体，并具备结构化动作能力。*

- **[x] 任务 2.1: 环境与动作函数**
    * **实现方案**: 创建一个`GraphEnvironment`类，封装`HierarchicalGraph`对象。在该类中实现智能体的所有动作函数，如 `semantic_anchor`, `expand_to_lca`, `query_node_details` 等。
    * **技术栈**: Python, `networkx`, ChromaDB Client.

- **[x] 任务 2.2: AH-RAG 智能体封装**
    * **实现方案**: 创建一个`AHRAG_Agent`类，其核心是"下一步动作决策"。提供规则回退与可选 LLM JSON 决策：接收 `GraphEnvironment` 观测，输出 `{action, params}`，并调用环境动作。使用统一LLM客户端进行智能体决策。
    * **技术栈**: Python Class, `ah_rag.utils.llm_client`。

- **[x] 任务 2.3: 状态与提示工程**
    * **实现方案**: 将 `GraphEnvironment` 观测裁剪为精简文本，提供严格 JSON 输出模式，并配套稳健解码：首轮温度 0.2、失败后降温到 0.0 与更强裁剪；仍失败回退规则策略。
    * **技术栈**: Prompt Engineering, JSON Schema。

#### **阶段三：推理引擎与流程打通 (Milestone 3: Interactive Reasoning Engine)**

*目标：将智能体与环境连接起来，形成一个完整的、可运行的单次推理流程。*

- **[x] 任务 3.1: 多轮交互循环**
    * **实现方案**: 实现 `InferenceEngine.run_inference(query, steps)`，在 3–5 步范围内执行"语义锚定→上卷/横向/下钻→证据归集→一次性答案生成"，并输出 `{answer, rationale, citations, used_actions, metrics}`。
    * **技术栈**: Python, `ah_rag.utils.llm_client`。

- **[x] 任务 3.2: 输入输出与日志记录**
    * **实现方案**: 统一 I/O 与事件日志，形成可复现、可审计、可度量的产物集：逐步事件流 `events.jsonl`、会话汇总 `summary.json`、答案产物 `answer.json`。
    * **技术栈**: `structlog`, `ah_rag.utils.config`。

- **[x] 任务 3.3: 答案生成（最终版：高保真模块）**
    * **实现方案**: 引入"上下文处理器 + 生成器"的双模块，实现预算可控、保真优先的答案生成。ContextProcessor对推理引擎收集的证据做优先级排序，AnswerGenerator采用自我验证提示与严格 JSON 抽取。
    * **技术栈**: tiktoken, `ah_rag.utils.llm_client`, `structlog`。

- **[x] 任务 3.4: 标准化基准评估框架**
    * **实现方案**: 基于Claude.md RAG诊断理论构建完整评估框架，核心洞察"RAG质量 = 检索器 × 生成器"。实现系统无关的通用评估，支持AH-RAG与Naive RAG等多系统对比。
    * **技术栈**: `datasets`, `evaluate`, `ah_rag.utils.llm_client`。

#### **阶段四：强化学习优化 (Milestone 4: RL Optimization)**

*目标：在冻结的知识环境上，通过RL优化智能体的检索与停步策略，兼顾准确率、faithfulness 与延迟/成本。*

- **[ ] 任务 4.1: 快照与契约冻结**
    * **实现方案**: 冻结 `graph/`、`vector_db/`、`artifacts/` 及 `meta.json`（包含 schema/prompt/model/seed/commit、vector_index 配置与 `graph_hash`）；导出 `requirements-lock`；CI 校验计数与 `graph_hash` 一致性。
    * **技术栈**: DVC/Git LFS、CI、`pip freeze`、本地 HF 缓存。
- **[x] 任务 4.2: 训练环境封装（Gym Wrapper）**
    * **实现方案**: 在 `GraphEnvironment` 之上提供 Gym 风格接口（obs、action、reward、done、info），固定长度观测（featurizer），显式动作集（禁用 auto-commit），支持并发 rollout。
    * **技术栈**: Python、`multiprocessing`/`ray`（可选）。
    * **落地产物**: `src/ah_rag/agent/gym_env.py`, `src/ah_rag/agent/featurizer.py`, `src/ah_rag/agent/reward.py`
    * **新增**: 动作掩码（无top节点仅允许结束）、重复动作惩罚（可配），并在PPO采样/评估中生效。
- **[x] 任务 4.3: 奖励函数设计**
    * **实现方案**: 采用“dense + terminal”组合奖励：
        - Dense（每步）：selection 增量、frontier 增量、步数惩罚（可加重复动作惩罚）；完全无LLM。
        - Terminal（收尾）：统一评估指标加权 `0.4·F1 + 0.3·faithfulness + 0.2·answer_relevancy + 0.1·contextual_recall`；训练期默认离线/小样本，避免在线Judge开销。
        - 支持课程学习调整（前期偏探索/召回，后期偏效率/faithfulness），奖励归一化。
    * **技术栈**: `evaluate`（终止）、无LLM密集塑形（步进）。
- **[ ] 任务 4.4: 算法与训练器（PPO→GRPO）**
    * **实现方案**: 首版 PPO（GAE、clip、entropy、value loss、梯度裁剪），8–32 并发 × T=8–12 截断，早停与最佳 checkpoint；状态=featurizer 向量，动作掩码（无frontier时屏蔽扩展类），禁用 auto-commit；先 BC 预热，再 PPO；后续引入 GRPO（偏好对来自双策略轨迹 + LLM-as-Judge）。
    * **技术栈**: PyTorch、`stable-baselines3`/`trl`、`peft`（LoRA 可选）。
    * **落地进展**: 
        - ✅ 完成BC预热脚手架（`scripts/collect_trajectories.py`, `scripts/train_bc.py`, `scripts/eval_rl_policy.py`）并通过小样本验证。
        - ✅ 集成最小化PPO训练器（`src/ah_rag/agent/policy_ppo.py`, `scripts/train_ppo.py`），支持“向量化收集（n_envs）+ 早停（patience, min_improve）”，能在冻结图上进行短轮训练并保存策略。
        - ✅ 评估脚本支持 `--ppo-model` 与 `--bc-model` 切换。
        - ⏭ 后续：加入动作掩码/重复动作惩罚、并发env与早停、接入SB3/TRL替换自研PPO；择机上线GRPO。
- **[x] 任务 4.5: 训练数据与并发配置**
    * **实现方案**: 准备 train/val/test 切分；并发env与随机种子管理；记录样本切片指标。已提供最小化脚手架：
    * **技术栈**: `datasets`、配置管理（YAML/Env）。
    * **落地产物**: 轨迹采集与评估脚本：`scripts/collect_trajectories.py`, `scripts/eval_rl_policy.py`；占位训练脚本：`scripts/train_rl.py`

运行示例：
```bash
# 1) 收集随机策略轨迹（用于BC预热或离线调试）
python3 scripts/collect_trajectories.py --dataset hotpotqa --limit 20 --out artifacts/rl/trajectories.jsonl

# 2) 训练占位策略（按动作频次生成先验）
python3 scripts/train_rl.py --traj artifacts/rl/trajectories.jsonl --out artifacts/rl/policy.json

# 3) 使用先验策略进行轻量评估（只看检索侧指标）
python3 scripts/eval_rl_policy.py --dataset hotpotqa --limit 5 --policy artifacts/rl/policy.json --out artifacts/rl/eval.json
```
- **[ ] 任务 4.6: 评估与晋级门槛**
    * **实现方案**:
        - 固定评估协议：固定温度/随机种子/数据切分（train/val/test），多次重跑（≥3 seeds）。
        - 对照基线：Naive RAG、启发式策略、随机策略、消融（禁用某类动作/权重固定）。
        - 指标维度：F1/EM、faithfulness、answer_relevancy、contextual_recall/precision、平均步数、expansions、p50/p95延迟、（可选）token/成本。
        - 晋级门槛（建议）：F1 ≥ baseline+5pt；faithfulness ≥ 0.75；平均步数不升；p95延迟 ≤ baseline+10%；Judge/评估失败率 < 2%。
        - 置信评估：bootstrap 95% 置信区间；统计显著性检验；run_id 绑定 graph_hash 以保证可复现。
        - 报告产出：JSON/CSV 持久化 + W&B/MLflow 报表；对失败样本自动汇总 primary_issue 分布与典型case。
    * **技术栈**: Weights & Biases/MLflow、`datasets`、统计检验（bootstrap）。
    * **落地进展**:
        - ✅ 新增评估门槛脚本：`scripts/eval_gate.py`，可集成CI（非零退出即不通过）。
        - ✅ Make 目标：`make rl-gate`（默认 HotpotQA limit=5）。
        - ⛳ 本轮 Gate 结果（limit=5）：
            - 先前（无PPO接入）：F1≈0.462，faith≈0.50，未达门槛。
            - 最新（PPO推理接入，`rl.inference.use_ppo=true`）：F1≈0.562，faith≈0.50，仍未达建议门槛（F1≥0.55, faith≥0.60）但F1已上升，下一步优先提升faithfulness。
        - ▶ 后续动作：继续PPO迭代（动作掩码/重复惩罚已开）、课程学习与BC数据扩充，优先提升faithfulness与F1。
- **[ ] 任务 4.7: 运行模式拨码（落地）**
    * **实现方案**: 研究/生产双模式；生产下限制步数、关闭昂贵动作与实时 Judge；异常熔断回退 vanilla RAG。
    * **技术栈**: 配置拨码、监控/熔断。
- **[ ] 任务 4.8: 监控与可观测性**
    * **实现方案**:
        - 事件与轨迹：规范化 `events.jsonl` 事件schema（步骤、动作、frontier/selection 规模、用时），按会话聚合沉淀。
        - 线上指标：LLM模块级 RPM/429率/重试与退避统计（来自 llm_client）、action 分布漂移、selection/frontier 规模分布、平均步数与延迟。
        - 漂移告警：与训练期分布做PSI/KS监测；阈值触发Slack/邮件；自动降级到启发式或vanilla RAG。
        - Dashboard：W&B面板或Grafana（Prometheus中转），含关键KPI与近7/30天趋势；run与graph_hash关联便于排障。
        - 取样留存：按比例采样会话（含上下文与决策），用于事后诊断与偏好标注（GRPO数据源）。
    * **技术栈**: 结构化日志、W&B/Grafana/Prometheus、报警通道（Slack/Email）。

---

### **LLM配置统一清理（2024-12-14 完成）**

**目标**：消除核心模块中的LLM调用冲突，统一使用`config/ah_rag.yaml`配置文件管理所有LLM设置。

**解决方案**：

1. **统一LLM客户端管理**（`src/ah_rag/utils/llm_client.py`）
   - 创建`LLMClientManager`类，集中管理所有模块的LLM配置
   - 定义`LLMModule`枚举：`KNOWLEDGE_EXTRACTION`, `SEMANTIC_AGGREGATION`, `AGENT_DECISION`, `ANSWER_GENERATION`, `EVALUATION_JUDGE`
   - 提供便捷函数：`is_llm_enabled()`, `get_llm_client()`, `create_chat_completion()`

2. **核心模块清理完成**：
   - **hypergraph_extractor.py**：使用`create_chat_completion(LLMModule.KNOWLEDGE_EXTRACTION)`，添加LLM禁用时的启发式回退
   - **semantic_aggregator.py**：所有摘要和评审调用统一使用`LLMModule.SEMANTIC_AGGREGATION`
   - **agent.py**：使用`LLMModule.AGENT_DECISION`进行智能体决策，保持规则回退机制
   - **answer_eval.py**：使用`LLMModule.EVALUATION_JUDGE`进行答案评估

3. **配置文件增强**（`config/ah_rag.yaml`）
   - 每个模块独立的LLM配置（provider, model, temperature等）
   - 全局和模块级别的开关控制
   - 支持多provider配置（kimi, qwen, deepseek, openai）

**验证结果**：
- ✅ 所有模块正常导入和运行
- ✅ 配置文件正确读取，开关机制生效
- ✅ LLM禁用时回退机制正常工作
- ✅ 端到端功能测试通过
- ✅ 消除了配置冲突和重复代码

---

### **项目实施概览**

#### Phase 1/4 — 知识环境（Extract / Aggregate / Graph）
- 配置：`config/ah_rag.yaml`
- 源码：`src/ah_rag/extract/`, `src/ah_rag/aggregate/`, `src/ah_rag/graph/`
- 产物：`artifacts/`, `graph/`, `vector_db/`
- 运行示例：`python3 scripts/demo_cli.py samples/hotpotqa_mini.txt`

#### Phase 2/4 — 智能体与推理（Agent / Inference / Answer）
- 源码：`src/ah_rag/agent/`, `src/ah_rag/answer/`, `src/ah_rag/utils/`
- 核心文件：`environment.py`, `agent.py`, `inference.py`, `context_processor.py`, `generator.py`, `llm_client.py`
- 运行示例：`python3 scripts/agent_cli.py "Scott Derrickson movies" --steps 2`

#### Phase 3/4 — 评估与诊断（Eval / Benchmark）
- 源码：`src/ah_rag/eval/answer_eval.py`, `scripts/run_benchmark.py`, `baselines/naive_rag.py`
- 运行示例：`python3 scripts/run_benchmark.py --dataset hotpotqa --system ah_rag --limit 10`

#### Phase 4/4 — 强化学习（RL）
- 计划中，含快照冻结、Gym封装、奖励设计、PPO/GRPO训练等8个子任务

### **项目依赖**

#### 核心依赖
- Python ≥ 3.10
- `sentence-transformers`（嵌入）
- `transformers`, `huggingface-hub`（模型与缓存）
- `chromadb`（向量索引）
- `networkx`, `numpy`
- `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`（聚类与降维）
- `datasets`, `evaluate`（数据与评估）
- `openai`（兼容 Kimi/DeepSeek/OpenAI API）
- `tiktoken`（token 预算）
- `pydantic>=2`, `python-dotenv`
- `requests`, `tqdm`

#### 可选/开发依赖
- `pytest`（单元测试）
- `mlflow`/`wandb`（实验跟踪）
- `dvc`（大文件/快照管理）

---

## **快速上手与测试指南**

### **环境准备**

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **配置LLM服务**
编辑 `config/ah_rag.yaml`，确保至少启用一个LLM提供商：
```yaml
llm:
  enabled: true
  default_model: "deepseek-chat"
  providers:
    kimi:
      enabled: true
      api_key: "YOUR_KIMI_API_KEY"
      base_url: "https://api.moonshot.cn/v1"
    deepseek:
      enabled: true
      api_key: "YOUR_DEEPSEEK_API_KEY"
      base_url: "https://api.deepseek.com"
```

3. **检查LLM配置**
```bash
# 验证LLM配置是否正确
python3 -c "
from ah_rag.utils.config import load_config
from ah_rag.utils.llm_client import is_llm_enabled, LLMModule

config = load_config()
print(f'LLM enabled globally = {config.get(\"llm\", {}).get(\"enabled\", False)}')
print(f'Knowledge extraction enabled = {is_llm_enabled(LLMModule.KNOWLEDGE_EXTRACTION)}')
print(f'Agent decision enabled = {is_llm_enabled(LLMModule.AGENT_DECISION)}')
"
```

### **完整流程测试**

#### **步骤1: 构建知识图谱**
```bash
# 使用示例文档构建知识图谱
python3 scripts/demo_cli.py samples/hotpotqa_mini.txt

# 验证构建结果
ls -la graph/        # 查看图结构文件
ls -la vector_db/    # 查看向量索引
ls -la artifacts/    # 查看中间产物
```

#### **步骤2: 测试各个模块**

**2.1 测试知识检索环境**
```bash
# 测试语义检索和图扩展
python3 scripts/env_cli.py "Scott Derrickson" --debug
python3 scripts/env_cli.py "quantum processor" --debug

# 测试自定义过滤器和权重
python3 scripts/env_cli.py "American director" \
  --filters "judge>=6" "type=entity" \
  --weights "alpha=0.6" "beta=0.25" "delta=0.15" \
  --expand parents
```

**2.2 测试智能体决策**
```bash
# 测试规则回退模式（无需LLM）
python3 scripts/agent_cli.py "Scott Derrickson movies" --steps 3

# 测试LLM决策模式
python3 scripts/agent_cli.py "quantum computing breakthrough" --steps 3 --llm --debug
```

**2.3 测试答案生成**
```bash
# 独立测试答案生成器
echo '["ent:scott_derrickson", "sum:topic_1"]' > evidence.json
python3 scripts/answer_cli.py "Who is Scott Derrickson?" --evidence evidence.json
```

#### **步骤3: 端到端基准测试**

**3.1 标准HotpotQA评估（推荐）**
```bash
# 快速测试 - AH-RAG系统，5条样本
python3 scripts/run_benchmark.py \
  --dataset hotpotqa \
  --system ah_rag \
  --limit 5 \
  --judge-sample 1.0 \
  --out reports/hotpotqa_quick_test.json

# 完整评估 - 对比AH-RAG vs Naive RAG，50条样本
python3 scripts/run_benchmark.py \
  --dataset hotpotqa \
  --system both \
  --limit 50 \
  --judge-sample 0.2 \
  --out reports/hotpotqa_comparison.json

# TriviaQA评估
python3 scripts/run_benchmark.py \
  --dataset triviaqa \
  --system ah_rag \
  --limit 20 \
  --out reports/triviaqa_test.json
```

**3.2 诊断分析**
```bash
# 查看详细评估报告
cat reports/hotpotqa_quick_test.json | jq '.aggregate'

# 查看单条样本详情
cat reports/hotpotqa_quick_test.json | jq '.items[0]'
```

### **常见问题排除**

#### **LLM配置问题**
```bash
# 检查API密钥是否正确
python3 -c "
from ah_rag.utils.llm_client import get_llm_manager
manager = get_llm_manager()
print('Available providers:', list(manager.providers.keys()))
"

# 若遇到429/限速，可在 config/ah_rag.yaml 调整模块级节流：
# llm.modules.knowledge_extraction.max_retries: 6
# llm.modules.knowledge_extraction.rate_limit_wait: 6
# llm.modules.knowledge_extraction.retry_jitter: 1.5
# llm.modules.semantic_aggregation.rate_limit_wait: 5

# 测试单个LLM模块
python3 -c "
from ah_rag.utils.llm_client import create_chat_completion, LLMModule
try:
    resp = create_chat_completion(
        LLMModule.KNOWLEDGE_EXTRACTION,
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=10
    )
    print('LLM test successful:', resp.choices[0].message.content)
except Exception as e:
    print('LLM test failed:', e)
"
```

#### **知识图谱构建问题**
```bash
# 检查图结构完整性
python3 -c "
from ah_rag.graph.hierarchical_graph import HierarchicalGraph
try:
    hg = HierarchicalGraph.load('graph')
    print(f'Graph loaded: {hg.node_count()} nodes, {hg.edge_count()} edges')
    print(f'Layers: {hg.get_layer_distribution()}')
except Exception as e:
    print('Graph load failed:', e)
    print('Please run: python3 scripts/demo_cli.py samples/hotpotqa_mini.txt')
"
```

#### **向量索引问题**
```bash
# 重建向量索引
python3 -c "
from ah_rag.graph.hierarchical_graph import HierarchicalGraph
hg = HierarchicalGraph.load('graph')
hg.build_vector_index('vector_db', layers={0,1,2})
print('Vector index rebuilt successfully')
"
```

### **性能基准参考**

#### **预期指标范围**
- **HotpotQA数据集**:
  - F1 Score: 0.4-0.7
  - EM Score: 0.2-0.5
  - Contextual Recall: 0.5-0.8
  - Faithfulness: 0.6-0.9
  - 平均推理步数: 3-5步

#### **系统对比**
- **AH-RAG vs Naive RAG**: AH-RAG在contextual_recall和faithfulness上通常表现更好
- **诊断分析**: 关注primary_issue分布，识别检索vs生成问题

### **开发调试**

#### **日志配置**
```bash
# 启用详细日志
export AH_RAG_LOG_LEVEL=debug
export AH_RAG_REDACT=false

# 查看推理过程日志
tail -f artifacts/phase2/*/events.jsonl
```

#### **配置文件模板**
复制并修改 `config/ah_rag.yaml` 中的关键参数：
- `llm.modules.*.enabled`: 控制各模块LLM使用
- `inference.steps`: 推理步数限制
- `evaluation.judge.sample_ratio`: Judge采样比例
- `answer.context.token_budget`: 上下文token预算
