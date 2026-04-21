from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 将对话历史 + 新问题 压缩为独立问题（用于历史感知检索）
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "根据对话历史和最新用户问题，将问题改写为一个独立、完整的问题，"
     "使其在没有对话历史的情况下也能被理解。"
     "如果问题本身已经完整，直接返回原问题，不要回答它。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 基于检索到的上下文生成答案
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个专业的学习助手。请根据以下检索到的上下文内容，准确、简洁地回答用户问题。\n"
     "如果上下文中没有足够信息，请直接说明'我在知识库中没有找到相关内容'，不要编造答案。\n\n"
     "上下文：\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
