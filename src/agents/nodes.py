"""
Agent 节点模块
负责：LangGraph 工作流中的所有节点定义

扩展指南：
- 添加新节点：定义新函数，然后在 graph.py 中注册
- 修改节点行为：直接编辑对应的节点函数
"""
from typing import List
from langchain_core.messages import AIMessage, HumanMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import extract_health_info, load_health_profile
from core.utils import detect_mode, grade_documents, rewrite_query
from config.settings import MAX_RETRIEVAL_LOOPS


def create_nodes(llm, llm_with_tools, vectorstore, web_search_tool, medical_tools_list):
    """
    创建所有节点函数
    
    Args:
        llm: 基础 LLM
        llm_with_tools: 带工具的 LLM
        vectorstore: 向量存储
        web_search_tool: 网络搜索工具
        medical_tools_list: 医学工具列表
    
    Returns:
        包含所有节点函数的字典
    """
    
    def router_node(state):
        """路由节点 - 分析问题并决定处理流程"""
        messages = state["messages"]
        user_id = state.get("user_id", "anonymous")
        question = messages[-1].content
        
        print(f"\n🧭 [分析问题中...]")
        
        # 提取健康信息（仅登录用户）
        if user_id and user_id != "anonymous":
            extract_health_info(question, user_id, llm)
        
        # 加载健康档案
        health_profile = load_health_profile(user_id) if user_id != "anonymous" else ""
        
        # 检测模式
        mode = detect_mode(question)
        print(f"  → {'健康评估' if mode == 'assessment' else '知识检索'}")
        
        return {
            "mode": mode,
            "need_tool": mode == "assessment",
            "need_rag": True,
            "need_web": False,
            "loop_step": 0,
            "documents": [],
            "used_web_search": False,
            "health_profile": health_profile,
            "summary": ""
        }
    
    def assessment_tool_node(state):
        """健康评估工具节点 - 调用计算工具"""
        print("📊 [计算健康指标...]")
        question = state["messages"][-1].content
        
        response = llm_with_tools.invoke(question)
        output = ""
        
        if response.tool_calls:
            results = []
            for call in response.tool_calls:
                tool = next((t for t in medical_tools_list if t.name == call["name"]), None)
                if tool:
                    try:
                        res = tool.invoke(call["args"])
                        results.append(f"📊 {str(res)}")
                    except Exception as e:
                        results.append(f"❌ 计算错误: {e}")
            output = "\n\n".join(results)
        else:
            output = "⚠️ 请提供具体数据，如 '我170cm，70kg，计算BMI'"
        
        return {"tool_output": output}
    
    def retrieve_node(state):
        """本地检索节点 - 从向量库检索"""
        print("📚 [检索知识库...]")
        question = state["messages"][-1].content
        
        search_query = f"{question} 健康建议" if state.get("tool_output") else question
        docs = vectorstore.similarity_search(search_query, k=4)
        doc_contents = [d.page_content for d in docs]
        
        return {"documents": doc_contents, "loop_step": state["loop_step"] + 1}
    
    def web_search_node(state):
        """Web搜索节点 - 联网搜索"""
        print("🌐 [联网搜索...]")
        question = state["messages"][-1].content
        
        try:
            results = web_search_tool.invoke({"query": question})
            web_contents = [res['content'] for res in results]
            return {"documents": web_contents, "used_web_search": True}
        except Exception as e:
            return {"documents": ["⚠️ 网络搜索暂时不可用"], "used_web_search": True}
    
    def grade_and_generate_node(state):
        """评分与生成节点 - 评估文档并生成回答"""
        question = state["messages"][-1].content
        docs = state["documents"]
        mode = state.get("mode", "science")
        health_profile = state.get("health_profile", "")
        
        score = grade_documents(question, docs, llm)
        
        if score == "yes":
            print("💡 [生成回答...]")
            context = "\n\n".join(docs)
            source_tag = "(来源: 互联网)" if state["used_web_search"] else "(来源: 医学知识库)"
            
            # 构建记忆上下文
            memory_context = ""
            if health_profile:
                memory_context = f"【用户健康档案】\n{health_profile}\n---\n"
            
            if mode == "assessment":
                tool_result = state.get("tool_output", "")
                prompt = f"""
你是专业的健康顾问。根据计算结果和医学知识，给出个性化建议。

{memory_context}
【评估结果】
{tool_result}

【参考资料】{source_tag}
{context}

【问题】{question}

请给出：1. 结果解读 2. 健康建议 3. 注意事项（特别注意过敏史和疾病史）
语气专业但亲切。
"""
            else:
                prompt = f"""
你是医学科普专家。用通俗易懂的语言回答。

{memory_context}
【参考资料】{source_tag}
{context}

【问题】{question}

要求：先简要回答，再展开解释，最后给出实用建议。
"""
            
            answer = llm.invoke(prompt).content
            return {"rag_output": answer, "final_answer": "ready"}
        
        elif state["loop_step"] >= MAX_RETRIEVAL_LOOPS:
            if not state["used_web_search"]:
                return {"final_answer": "go_web"}
            else:
                context = "\n\n".join(docs)
                prompt = f"根据有限信息尽力回答：\n资料：{context}\n问题：{question}"
                answer = llm.invoke(prompt).content
                return {"rag_output": answer, "final_answer": "ready"}
        else:
            new_query = rewrite_query(question, llm)
            return {"messages": [HumanMessage(content=new_query)]}
    
    def summarizer_node(state):
        """总结节点 - 格式化最终输出"""
        mode = state.get("mode", "science")
        tool_output = state.get("tool_output", "")
        rag_output = state.get("rag_output", "")
        health_profile = state.get("health_profile", "")
        
        profile_note = "\n📋 已参考你的健康档案" if health_profile else ""
        
        if mode == "assessment" and tool_output:
            final_text = f"""
{'═' * 50}
📊 健康评估结果
{'═' * 50}

{tool_output}

{'─' * 50}
💡 建议
{'─' * 50}

{rag_output if rag_output else '暂无额外建议'}{profile_note}

⚠️ 以上仅供参考，具体请咨询医生。
"""
        else:
            final_text = f"""
{'═' * 50}
📖 回答
{'═' * 50}

{rag_output if rag_output else '抱歉，暂时无法找到相关信息。'}{profile_note}

💡 以上信息仅供科普学习，具体请遵医嘱。
"""
        
        return {"final_answer": final_text, "messages": [AIMessage(content=final_text)]}
    
    return {
        "router": router_node,
        "assessment_tool": assessment_tool_node,
        "retrieve": retrieve_node,
        "web_search": web_search_node,
        "grade_loop": grade_and_generate_node,
        "summarizer": summarizer_node
    }
