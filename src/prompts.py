"""
Prompt 模板模块
───────────────
定义两个 Prompt：

1. CONDENSE_PROMPT（问题压缩 Prompt）
   用于多轮对话场景。
   问题：用户的追问通常是简短的指代（"它有什么缺点？"），
         如果直接拿去检索，找不到相关内容，因为向量库里没有"它"。
   解决：先把对话历史 + 新问题 交给 LLM，改写成一个独立完整的问题
         （"逻辑回归有什么缺点？"），再拿去检索。

2. QA_PROMPT（问答 Prompt）
   把检索到的上下文和用户问题一起交给 LLM，要求它基于上下文回答，
   不知道就直说，不要编造——这是 RAG 防幻觉的关键。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ── Prompt 1：将多轮对话中的追问改写为独立问题 ────────────────────────────────
#
# 模板变量说明：
#   {chat_history} — MessagesPlaceholder，会被替换为 LangChain Message 列表
#                    （HumanMessage / AIMessage 交替排列）
#   {input}        — 用户的最新问题
#
# 输出：改写后的独立问题字符串（不是答案，只是问题）
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "根据对话历史和最新用户问题，将问题改写为一个独立、完整的问题，"
     "使其在没有对话历史的情况下也能被理解。"
     "如果问题本身已经完整，直接返回原问题，不要回答它。"),
    MessagesPlaceholder("chat_history"),  # 这里会被实际的对话历史填充
    ("human", "{input}"),                 # 用户的最新输入
])


# ── Prompt 2：基于检索到的上下文生成答案 ─────────────────────────────────────
#
# 模板变量说明：
#   {context}      — 从向量库检索到的最相关文本块（拼接成字符串）
#   {chat_history} — 对话历史（让 LLM 理解上下文，保持对话连贯）
#   {input}        — 用户的当前问题
#
# 系统提示的关键设计：
#   - "根据上下文回答" → 防止 LLM 用训练数据编造答案
#   - "没有就说没有"   → 防止幻觉，提高可信度
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的学习助手。请根据以下检索到的上下文内容，准确、简洁地回答用户问题。\n"
     "如果上下文中没有足够信息，请直接说明'我在知识库中没有找到相关内容'，不要编造答案。\n\n"
     "上下文：\n{context}"),
    MessagesPlaceholder("chat_history"),  # 对话历史，保持多轮连贯
    ("human", "{input}"),
])
