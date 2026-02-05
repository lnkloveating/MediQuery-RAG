# 🏥 MediQuery-RAG 科普医疗智能助手

基于 **LangGraph + RAG + 多轮追问** 的智能医疗问诊系统

---

## 📋 目录

- [项目简介](#项目简介)
- [技术架构](#技术架构)
- [开发历程](#开发历程)
- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [安装与运行](#安装与运行)
- [使用演示](#使用演示)
- [技术细节](#技术细节)

---

## 项目简介

MediQuery-RAG 是一个基于大语言模型的智能医疗问诊系统,旨在提供:

- 🩺 **结构化问诊**: 引导式收集用户健康信息
- 🤖 **智能追问**: AI 根据症状自动追问关键信息(位置、性质、持续时间等)
- 📊 **健康评估**: 自动计算 BMI/BMR,AI 分析身体状况
- ⚠️ **风险预警**: 实时检测高危症状,紧急情况立即提醒就医
- 📚 **RAG 知识库**: 结合医学知识库和联网搜索生成建议
- 💾 **用户档案**: 持久化存储用户健康档案和问诊历史

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面层                            │
│                    (CLI / Gradio Web)                       │
├─────────────────────────────────────────────────────────────┤
│                      业务逻辑层                              │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  结构化问诊模块   │  │   科普问答模块   │                  │
│  │ (多轮追问/风险评估)│  │  (自由对话)     │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           └──────────┬─────────┘                            │
│                      ▼                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LangGraph 工作流引擎                     │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │   │
│  │  │Router│→ │Retrieve│→│Grade │→ │WebSrc│→ │Summary│ │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                       数据层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ ChromaDB    │  │ SQLite      │  │ JSON Files  │         │
│  │ (向量知识库) │  │ (对话历史)   │  │ (用户档案)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                       模型层                                 │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Qwen2.5:7B      │  │ dmeta-embedding │                  │
│  │ (对话/推理)      │  │ (中文向量化)     │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 核心设计理念

本项目基于 LangGraph 状态机框架,重点解决传统 RAG 机器人"聊完就忘"或"上下文过长导致幻觉"的问题。

- **用户关键信息持久化**: 系统会自动抓取对话中的关键实体(如患病名称、过敏源、身高体重),存入长期记忆。
- **Thread ID 索引**: 只要提供相同的 thread_id(或用户 ID),无论何时打开,系统都能瞬间回忆起你的健康档案。
- **对话细节自动修剪 (Trim)**: 单次问诊结束或开启新话题时,系统会自动清理琐碎的对话细节(短期记忆),只保留核心结论。这既节省了 Token,又保证了长期记忆的纯净度。

### 双层记忆体架构 (核心)

这是本项目最核心的技术实现,我们将记忆分为两层管理:

#### 1. 长期记忆 (Long-term Memory)

**存储内容**: 关键健康画像(Profile)。包括:身高、体重、BMI、过敏史、慢性病史、家族病史等。

**实现方式**:

- **提取**: 在对话过程中,使用 LLM 动态抓取关键信息。
- **存储**: 持久化为标准 JSON 文件 (`user_data/{id}/profile.json`)。
- **可视化**: 支持自动导出为 Markdown 病历卡 (`history.md`),方便用户阅读。
- **生命周期**: 永久保存。每次对话开始前,系统会自动加载该档案作为 System Prompt 的一部分。

#### 2. 短期记忆 (Short-term Memory)

**存储内容**: 当前问诊的对话细节。例如:"肚子疼是绞痛还是胀痛"、"持续了3天还是5天"、"昨晚吃了什么"等上下文细节。

**实现方式**:

- **LangGraph State**: 利用 LangGraph 的 Checkpointer 机制,将对话状态 (messages 列表) 暂存在 SQLite/内存中。
- **多轮对话**: 在同一个 Session 内,模型能完美回忆起你刚才说的每一个细节,实现深度追问。
- **生命周期**: 会话级 (Session-based)。
  - 当一次问诊结束(或用户主动发起新问诊)时,系统会执行 Trim (修剪) 操作。
  - **销毁**: 琐碎的对话流被丢弃。
  - **归档**: 对话的结论(如"诊断为急性肠胃炎")被提取并转存入长期记忆。

### 主要功能

- 🧠 **智能记忆管理**:
  - **自动提取**: 你说"我对青霉素过敏",系统自动更新长期档案。
  - **跨时空记忆**: 哪怕过了一个月再来,系统依然记得你的过敏史,并在生成建议时自动规避相关药物。

- 🩺 **结构化问诊流程**:
  - 基于 LangGraph 的图结构控制,引导用户分阶段完成:基础信息 → 病史采集 → 意图确认 → 症状描述。
  - 避免大模型"随心所欲"地乱问。

- 📊 **自动健康画像**:
  - 后台静默计算 BMI / BMR / 理想体重。
  - 结合长期记忆中的数据,为用户生成实时的身体状况评估。

- 🚨 **风险分级预警**:
  - 实时监控对话内容,识别高危关键词(如自杀倾向、剧烈胸痛等),触发硬规则拦截,优先保障用户安全。

---

## 开发历程

### 🔍 第一阶段: 嵌入模型选型

项目初期面临的核心问题是**中文语义理解**。我们测试了多个嵌入模型:

| 模型 | 问题 |
|------|------|
| `nomic-embed-text` | 对中文支持差,语义相似度计算不准确 |
| `bge-m3` | 效果一般,检索召回率不理想 |
| `mxbai-embed-large` | 英文效果好,中文效果差 |
| **`shaw/dmeta-embedding-zh`** | ✅ 专门针对中文优化,语义理解准确 |

**最终选择**: `shaw/dmeta-embedding-zh`

```python
# src/medical_engine.py
embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")
```

### 🏗️ 第二阶段: LangGraph 工作流搭建

使用 LangGraph 构建 **Self-RAG** 工作流,实现智能检索和生成:

```
START → Router → Retrieve → Grade → [通过] → Summarizer → END
                    ↑         │
                    │    [不通过]
                    │         ↓
                    ←─── Rewrite Query
                              │
                         [超过3次]
                              ↓
                         Web Search
```

### 🩺 第三阶段: 结构化问诊系统

从简单的问答升级为**结构化问诊流程**:

1. **用户识别**: 手机号登录,老用户恢复档案
2. **基础信息采集**: 性别、年龄、身高、体重
3. **病史信息采集**: 家族病史、过敏史、慢性病、用药情况
4. **咨询目的选择**: 健康管理 / 身体不适
5. **症状描述 + AI 追问**: 智能收集症状详情
6. **风险评估**: AI 判断紧急程度
7. **RAG 生成建议**: 结合知识库和联网搜索

### 🤖 第四阶段: 多轮智能追问

实现了基于 **LangChain Messages** 的对话记忆:

```python
messages = [
    SystemMessage(content="你是问诊医生..."),
    HumanMessage(content="我肚子疼"),
    AIMessage(content="肚子哪个位置疼?"),
    HumanMessage(content="下腹部"),
    AIMessage(content="是什么样的疼法?"),
    HumanMessage(content="绞痛"),
    # AI 能看到完整对话历史,不会重复提问
]
```

### 📜 第五阶段: 历史摘要注入

实现用户历史问诊记录的注入,让系统能够:

- 识别相似症状:"您上周也咨询过头痛"
- 结合历史给出建议:"考虑到您多次出现类似症状..."

---

## 核心功能

### 1. 🩺 智能健康问诊

- 结构化问诊流程,引导用户完成信息填写
- AI 自动追问关键信息(位置、性质、持续时间、诱因、伴随症状)
- 最多追问 3 轮,避免用户疲劳
- 实时风险评估,高危症状立即预警

### 2. 📊 健康指标计算

- **BMI**: 身体质量指数
- **BMR**: 基础代谢率
- **理想体重**: 根据身高性别计算
- **AI 评估**: 综合分析身体状况

### 3. ⚠️ 风险评估系统

| 等级 | 触发条件 | 响应 |
|------|----------|------|
| CRITICAL | 自杀/自残关键词 | 立即终止,提供心理援助热线 |
| HIGH | AI 判断需紧急就医 | 强烈建议 24 小时内就医 |
| MEDIUM | 严重程度 ≥7 分或中等风险关键词 | 建议近期就医 |
| LOW | 其他情况 | 提供健康建议 |

### 4. 📚 RAG 知识检索

- **本地知识库**: ChromaDB 向量存储医学科普内容
- **联网搜索**: Tavily API 获取最新信息
- **Self-RAG**: 自动评估检索质量,必要时重写查询或联网搜索

### 5. 💾 用户档案管理

- 手机号哈希作为用户 ID
- JSON 格式存储用户档案
- 问诊记录按会话保存
- 自动生成 Markdown 格式的健康档案

---

## 项目结构

```
MediQuery-RAG/
├── main.py                     # 程序入口
├── .env                        # 环境变量 (TAVILY_API_KEY)
├── medical_db/                 # ChromaDB 向量数据库
├── user_data/                  # 用户档案存储
│   └── {user_id}/
│       ├── profile.json        # 用户基础信息
│       ├── sessions/           # 问诊记录
│       │   └── 20260205_xxxx.json
│       └── history.md          # Markdown 格式档案
│
├── config/
│   └── settings.py             # 全局配置
│
├── data/
│   └── medical_data.txt        # 医学知识库原始数据
│
└── src/
    ├── medical_engine.py       # 核心引擎: LLM/向量库/搜索工具初始化
    ├── ingest_medical.py       # 知识库入库脚本
    ├── tools.py                # 医学计算工具 (BMI/BMR/理想体重)
    │
    ├── agents/                 # LangGraph 工作流
    │   ├── __init__.py
    │   ├── graph.py            # 工作流定义和编译
    │   └── nodes.py            # 各节点实现
    │
    ├── consultation/           # 结构化问诊模块
    │   ├── __init__.py
    │   └── structured_consultation.py  # 问诊核心逻辑
    │
    ├── memory/                 # 记忆模块
    │   ├── __init__.py
    │   ├── health_extractor.py # 健康信息提取
    │   ├── profile_store.py    # 档案存储
    │   └── summary.py          # 摘要生成
    │
    ├── core/                   # 工具函数
    │   └── utils.py            # 模式检测/文档评分/查询重写
    │
    └── ui/                     # 用户界面
        └── interface.py        # CLI 界面实现
```

---

## 各模块功能说明

### `src/medical_engine.py` - 核心引擎

```python
# 初始化嵌入模型(中文优化)
embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh")

# 初始化对话模型
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# 初始化向量数据库
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# 初始化联网搜索
web_search_tool = TavilySearch(max_results=3)
```

### `src/agents/graph.py` - LangGraph 工作流

```python
def build_graph(nodes: dict):
    workflow = StateGraph(MedicalState)
    
    # 注册节点
    workflow.add_node("router", nodes["router"])
    workflow.add_node("retrieve", nodes["retrieve"])
    workflow.add_node("grade_loop", nodes["grade_loop"])
    workflow.add_node("web_search", nodes["web_search"])
    workflow.add_node("summarizer", nodes["summarizer"])
    
    # 定义流程
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router", route_after_router)
    workflow.add_edge("retrieve", "grade_loop")
    workflow.add_conditional_edges("grade_loop", route_self_rag)
    workflow.add_edge("web_search", "grade_loop")
    workflow.add_edge("summarizer", END)
    
    return workflow.compile(checkpointer=memory)
```

### `src/agents/nodes.py` - 节点实现

| 节点 | 功能 |
|------|------|
| `router_node` | 分析问题类型,决定处理流程 |
| `retrieve_node` | 从向量库检索相关文档 |
| `grade_and_generate_node` | 评估文档质量,生成回答或触发重试 |
| `web_search_node` | 联网搜索补充信息 |
| `summarizer_node` | 格式化最终输出 |

### `src/consultation/structured_consultation.py` - 问诊核心

```python
class StructuredConsultation:
    def identify_user(identifier)      # 用户识别
    def start_session()                # 开始问诊会话
    def get_current_question()         # 获取当前问题
    def process_answer(answer)         # 处理用户回答
    def _check_need_followup()         # AI 判断是否需要追问
    def _assess_risk_realtime(text)    # 实时风险评估
    def get_consultation_summary()     # 获取问诊摘要
    def get_history_summary()          # 获取历史摘要
    def save_session()                 # 保存会话
    def generate_history_markdown()    # 生成 Markdown 档案
```

### `src/ui/interface.py` - 用户界面

```python
def main_menu()                    # 主菜单
def structured_consultation_flow() # 结构化问诊流程
def free_qa_mode()                 # 自由问答模式
def _build_rag_query(summary)      # 构建 RAG 查询
```

### `src/tools.py` - 医学计算工具

```python
@tool
def calculate_bmi(height_cm, weight_kg):
    """计算 BMI 指数"""
    
@tool
def calculate_bmr(weight_kg, height_cm, age, gender):
    """计算基础代谢率"""
    
@tool
def calculate_ideal_weight(height_cm, gender):
    """计算理想体重"""
```

---

## 安装与运行

### 环境准备

请确保你的电脑已安装:

- Python 3.10+
- Ollama (用于运行本地大模型)
- Git

### 1. 克隆项目

```bash
git clone https://github.com/your-username/mediquery-rag.git
cd mediquery-rag
```

### 2. 创建虚拟环境 (推荐)

```bash
python -m venv venv
```

**Windows 激活**:
```bash
venv\Scripts\activate
```

**Mac/Linux 激活**:
```bash
source venv/bin/activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> 注意: 如果还没有 `requirements.txt`,请先安装核心库:
> ```bash
> pip install langchain langchain-community langchain-ollama langchain-chroma langgraph langgraph-checkpoint-sqlite chromadb tavily-python
> ```

### 4. 配置 Ollama 模型

本项目依赖 Ollama 运行 LLM 和 Embedding 模型。请确保 Ollama 软件已启动,并在终端运行:

```bash
# 下载主模型 (对话与逻辑推理)
ollama pull qwen2.5:7b

# 下载 Embedding 模型 (用于知识库向量化,中文效果好)
ollama pull shaw/dmeta-embedding-zh
```

### 5. 配置 API Key (可选,用于联网搜索)

如果你需要启用联网搜索功能(当本地知识库回答不了时),需要配置 Tavily API:

1. 在项目根目录创建一个名为 `.env` 的文件。
2. 写入以下内容:
   ```
   TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxx
   ```

### 6. 知识库构建

这是最关键的一步。你需要把你的医学资料"喂"给系统,生成向量数据库。

**第一步: 准备数据**

将你的医学科普文章、TXT 文档、整理好的问答对,放入 `data/` 文件夹中。

- 默认文件: `data/medical_data.txt`

**第二步: 运行入库脚本**

运行以下命令,系统会自动读取 `data/` 下的文件,将其切片、向量化,并存入 `medical_db/` 文件夹。

```bash
python src/ingest_medical.py
```

### 7. 启动程序

```bash
python main.py
```

---

## 使用演示

### 启动界面

```
⚙️ 正在初始化医学引擎 (LLM & VectorStore)...
✅ Tavily 联网搜索已启用

╔══════════════════════════════════════════════════════════╗
║              🏥 科普医疗智能助手                          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   请选择服务模式:                                        ║
║                                                          ║
║   [1] 🩺 智能健康问诊(推荐)                             ║
║   [2] 📚 医学科普问答                                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

### 智能问诊流程

```
📱 您的手机号: 15166676667

👋 欢迎回来!
   档案ID: e9ac2f13...
   
📋 您的已有档案:
   ├── 性别: 男
   ├── 年龄: 19岁
   ├── 身高: 175.0cm | 体重: 75.0kg
   ├── BMI: 24.5
   └── 无已知慢性病

【问题 1】请问您今天咨询的目的是?
   1. 健康管理建议
   2. 身体不适咨询

👤 您的回答: 2

【问题 2】请简单描述一下您哪里不舒服?

👤 您的回答: 肚子疼
  🤔 [AI正在判断是否需要追问...]
  💡 [追问原因: 需要确定疼痛位置]

【问题 3】肚子哪个位置疼?
   1. 上腹部
   2. 下腹部
   3. 肚脐周围
   4. 整个肚子

👤 您的回答: 2
  🤔 [AI正在判断是否需要追问...]
  💡 [追问原因: 需要了解疼痛性质]

【问题 4】是什么样的疼法?
   1. 绞痛
   2. 胀痛
   3. 隐痛
   4. 刺痛

👤 您的回答: 1
  ✅ [信息已足够,无需追问]

【问题 5】严重程度打几分? (1-10)

👤 您的回答: 6

📊 评估结果
   ├── 主诉: 肚子疼
   ├── 持续时间: -
   ├── 严重程度: 6.0/10
   └── 风险等级: LOW

📚 [知识库] 检索到 5 条相关内容
🌐 [联网搜索] 找到 3 条结果
💡 [正在生成建议...]

══════════════════════════════════════════════════
📖 回答
══════════════════════════════════════════════════

根据您的描述,下腹部绞痛可能的原因包括...

[详细的健康建议]

📋 已参考你的健康档案
💡 以上信息仅供科普学习,具体请遵医嘱。
```

---

## 技术细节

### LangGraph 在项目中的应用

#### 1. 状态定义

```python
class MedicalState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史
    mode: str                                 # 模式: assessment/science
    user_id: str                              # 用户 ID
    documents: List[str]                      # 检索到的文档
    loop_step: int                            # 循环次数
    used_web_search: bool                     # 是否已联网搜索
    health_profile: str                       # 用户健康档案
    final_answer: str                         # 最终答案
```

#### 2. 条件路由

```python
def route_self_rag(state):
    decision = state.get("final_answer")
    if decision == "ready":
        return "summarizer"      # 生成完成
    elif decision == "go_web":
        return "web_search"      # 需要联网搜索
    return "retrieve"            # 重新检索
```

#### 3. 检查点持久化

```python
conn = sqlite3.connect(CHAT_HISTORY_DB)
memory = SqliteSaver(conn)
app = workflow.compile(checkpointer=memory)
```

### 多轮追问的消息记忆

```python
def _check_need_followup(self):
    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)]
    
    # 添加主诉
    messages.append(HumanMessage(content=f"我的症状是: {chief_complaint}"))
    
    # 添加历史追问
    for qa in session.followup_qa:
        messages.append(AIMessage(content=qa["question"]))
        messages.append(HumanMessage(content=qa["answer"]))
    
    # AI 判断是否继续追问
    messages.append(HumanMessage(content="请判断是否需要继续追问"))
    
    response = self.llm.invoke(messages)
```

---

## 依赖说明

```
langchain>=0.2.0
langchain-ollama>=0.1.0
langchain-chroma>=0.1.0
langchain-tavily>=0.1.0
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=1.0.0
chromadb>=0.4.0
python-dotenv>=1.0.0
```

---

## 优化空间与未来展望

虽然本项目实现了一个完整的"问诊-检索-回答"闭环,但作为一个 Demo 级别的系统,在面对真实复杂的医疗场景时仍存在局限性。以下是针对当前痛点的深度分析与改进路线图。

### 交互体验升级 (UI & 多模态)

- **Web 可视化界面**: 计划迁移至 Streamlit 或 Gradio 构建 Web 界面,提供更直观的聊天气泡体验。
- **语音问诊**: 集成 OpenAI Whisper 模型,支持老年人通过语音描述病情,系统自动转录为文字。
- **图片识别**: 集成 Vision 模型(如 LLaVA 或 GPT-4o-Vision),允许用户上传"化验单"、"药品说明书"或"皮肤患处照片",辅助 AI 进行更精准的判断。

### 解决"答非所问": 检索与知识库增强

**当前痛点**:

- **数据源匮乏**: 当前的数据库仅包含少量示例数据,无法覆盖复杂的医学问题。
- **检索精度不足**: 单纯的向量相似度检索(Dense Retrieval)容易忽略精确的医学专有名词(如药品化学名),导致检索回来的内容虽语义相关但非用户所需。

**优化方案**:

- **构建权威知识库**: 接入权威医学指南(药品说明书数据库、三甲医院科普文章),构建百万级的专业医疗知识库。
- **混合检索 (Hybrid Search)**: 引入 BM25 关键词检索 + 向量检索。BM25 能精准匹配专有名词(解决"抓不到关键词"的问题),向量检索负责理解语义。
- **重排序 (Re-ranking)**: 在检索结果和 LLM 之间增加一个 Cross-Encoder 重排序模型(如 bge-reranker),将最相关的文档排在最前面,防止无关信息干扰大模型。

### 提升大模型能力

- **更换更强基座**: 目前的 Qwen2.5:7b 虽然在本地运行流畅,但逻辑推理能力有限。未来可接入 DeepSeek-V3、GPT-4o 或本地部署 Qwen2.5-72B (需更高显存),以获得接近真人的诊断逻辑。
- **微调 (Fine-tuning)**: 如果数据量足够大,可使用真实的"医患对话数据集"对模型进行指令微调 (SFT),让模型学会像医生一样说话,而不是像百科全书一样堆砌知识。

### 与真实的医疗诊所实现合作

- **挂号推荐**: 结合用户的地理位置(Location),在给出建议后,自动推荐附近的医院或对应科室。
- **电子病历导出**: 支持将问诊记录一键生成 PDF 报告,方便用户就医时直接出示给医生。

---

## 免责声明

⚠️ **本系统仅供健康科普参考,不能替代专业医生诊断。如有身体不适,请及时就医。**

---

## 致谢

- [LangChain](https://langchain.com/) - LLM 应用框架
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 工作流引擎
- [Ollama](https://ollama.com/) - 本地模型运行
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [Tavily](https://tavily.com/) - 联网搜索 API

---

## License

MIT License