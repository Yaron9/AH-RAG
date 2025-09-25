# ADT: L4 数字基座层（Context‑as‑a‑Service）路线与方案

## 1. 目标与范围（Vision & Scope）
- 作为系统“长时记忆”和“知识中心”，为上层智能体与交互系统提供统一、低延迟、可追溯的上下文服务。
- 面向多源异构（文档/结构/时序/事件）数据，支持时空检索、语义拼装、版本/快照与订阅推送。
- 以契约化输入与可复现快照保障训练/评估/生产的一致性；以监控与拨码满足延迟/成本目标。

## 2. L4 必备核心能力（Context‑as‑a‑Service）
- 数据平面与契约
  - 多源接入：文件/DB/消息（JDBC/CDC、S3、Kafka、MQTT/OPC‑UA）。
  - Schema Registry（Avro/Protobuf）+ 版本化数据契约，CI 校验契约一致。
  - 实时流处理：Flink/Kafka Streams（窗口/Join/乱序迟到处理/Exactly‑once）。
  - 元数据与血缘：数据目录、质量规则（缺失/越界/漂移）与告警。
- 知识图谱（时空增强）
  - 统一本体：设备/工况/事件/工序/人员/地理/文档；实体解歧与 ID 对齐。
  - 时态建模：节点/边携带有效时间（valid_from/valid_to），支持 bitemporal 与 as_of 查询。
  - 事件图：时序信号→事件节点（越界/波形/告警），挂接因果与工单/日志/文档。
- 检索与上下文构建
  - 混合检索：文本/结构/向量/时序统一查询；时空过滤（实体×时间窗×空间范围）。
  - 上下文组装：“骨架+细节”拼装与压缩（证据选取/预算控制）；可订阅的上下文刷新。
- 存储与服务
  - 分层存储：时序（Timescale/Influx）、向量（PGVector/Weaviate/Milvus）、文档（对象桶）、图（Neo4j/PG JSONB）。
  - 统一 API：GraphQL/gRPC/REST；Context DSL（实体/关系/时间窗/指标/阈值）；鉴权（RBAC/ABAC）。
- 可观测与治理
  - SLO：P95 延迟/上下文体积/成本预算；审计（谁在何时用过何上下文）。
  - 安全合规：多租户隔离、标签/脱敏、留存策略与 GDPR/合规对齐。

## 3. 时序传感数据能力（重点）
- 采集：MQTT/OPC‑UA → Kafka（带 Schema Registry）；设备侧时钟/时区策略（NTP/PTP）。
- 质量：缺测/离群/漂移/卡死检测，配置规则生成事件节点；质量位与单位校准。
- 表征：窗口统计、频域/形状特征，必要时 TS embedding（TS2Vec/TimesNet）入库，支持时序向量检索。
- 语义对齐：点位→业务语义表（温度/振动/流量…）、阈值与报警策略版本化。
- RAG 上下文：`as_of/between` 时间语义；“最近 k 个事件 + 近 T 统计摘要 + SOP 摘要”拼装。

## 4. 当前系统能力（AH‑RAG 现状）
- 抽取/聚合/建图：L0 超图 + L1/L2 摘要；层级图（NetworkX）与向量索引（Chroma）。
- 检索与上下文：混合检索、上下文预算控制与证据拼装（ContextProcessor）。
- 智能体：GraphEnvironment + AHRAG_Agent + InferenceEngine，策略式检索（semantic_anchor/expand/commit）。
- 评估闭环：组件指标 + LLM‑as‑Judge；F1/EM、faithfulness、answer_relevancy；诊断维度 `retriever/generator/both`。
- 契约化基础：`meta.json` 记录索引参数；快照意识与可复现评估；RL 训练前置条件基本具备。

## 5. 关键差距（与 L4 CaaS）
- 时序与实时：缺少 IoT/Kafka 接入、时序库、窗口与水位线处理、实时事件生成。
- 图谱时态：节点/边缺少有效时间/事务时间；无 bitemporal 与 as_of 查询。
- API 服务化：缺少统一 Context API、订阅/推送、SLO/熔断/拨码治理。
- 契约与治理：缺少 Schema Registry、血缘与质量监控、快照元数据标准化。
- 生产级存储：向量/图库的可扩与高可用、备份/恢复；多租户与安全隔离。

## 6. 升级路线（可执行）
### Phase A（2–4 周）：打通“时序 + 契约 + API”
- 数据面：部署 Kafka + Schema Registry；选 2–3 类关键点位接入（MQTT/OPC‑UA 网关）。
- 时序库：TimescaleDB/InfluxDB 建表（measurement/tags/fields/ts）；Flink/KStreams 生成事件节点并入图。
- 时态图谱：为实体/关系增 `valid_from/valid_to`；事件节点与实体/文档挂接；基本 as_of 查询。
- Context API：GraphQL/gRPC 的 `get_context(entity_ids, time_window, filters, budget)`；返回骨架+明细+统计。
- 契约与快照：冻结 `schema_version/prompt_version/model_revision/seed/commit`，输出 `graph_hash`；CI 做契约/计数/哈希校验。

### Phase B（4–8 周）：鲁棒检索 + 低延迟落地
- 混合检索：为时序段生成特征/embedding 并入索引；Context DSL 增 `as_of/between` 与空间过滤。
- 缓存与降级：Context 结果缓存（Redis/KV）；预算拨码（步数/扩展开关/压缩比例）；Judge 采样 + 异步离线评审。
- 监控与 SLO：端到端延迟、上下文体积、token 成本、失败率、数据质量；异常熔断回退 vanilla 流程。

### Phase C（并行）：策略优化与治理
- RL 策略：在冻结快照上训练（PPO/GRPO），奖励含“时序召回/事件覆盖”；生产拨码控制步数与成本。
- 治理与安全：多租户、权限与审计；留存策略与合规；配置与规则审计。

## 7. 成功度量（门槛与目标）
- 召回与质量：contextual_recall ≥ 0.95、contextual_relevancy ≥ 0.75；faithfulness ≥ 0.75；F1/EM ≥ 基线 + 5pt。
- 体验与成本：P95 延迟 ≤ 目标阈值（例如 ≤ 3–5s/查询），上下文 token ≤ 预算；Judge 采样 ≤ 20%。
- 稳定性：契约/快照 CI 通过；as_of 查询一致；故障回退成功率 ≥ 99.9%。

## 8. API 草案（示例）
### get_context（GraphQL/gRPC 伪接口）
- 入参：
  - `entity_ids: [string]`
  - `time_window: {from: ISO8601, to: ISO8601}`（支持 `as_of`）
  - `filters: {layers, types, spatial, metrics, events}`
  - `budget: {tokens: int, latency_ms: int}`
- 出参：
  - `skeleton: [node_id/title/summary/type/time_bounds]`
  - `details: [node_id/text/snippet]`
  - `timeseries: [metric, window, stats, anomalies]`
  - `events: [event_id/type/severity/start/end/links]`
  - `meta: {build_id, snapshot, cost, latency_ms}`

## 9. 技术栈建议
- Kafka/Schema Registry；Flink/Kafka Streams；Timescale/Influx。
- 图/向量：Neo4j 或 PG(JSONB/GIN) + PGVector / Weaviate / Milvus。
- API：GraphQL/gRPC/REST；Auth：RBAC/ABAC；缓存：Redis/KV。
- 评估与监控：Prometheus/Grafana，W&B/MLflow；CI：契约与快照校验。

## 10. 维护与 CI（防漂移）
- Champion–Challenger 抽取/Prompt 对照；覆盖率准入门槛（关键词/槽位/节点数）。
- 快照元数据：`meta.json` 记录 schema/prompt/model/revision/seed/commit、`graph_hash`；CI 比对计数/哈希。
- Overlay 回补：允许对个别样本增量补抽，不破坏基线；训练期总是使用冻结快照。

---
（本文件用于指引 L4 数字基座层的建设与落地，结合 AH‑RAG 作为知识中心与策略检索核心；可按 Phase A→B→C 执行并在里程碑上做复盘与拨码优化。）

