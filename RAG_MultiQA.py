import sys
import os
import time
import json
import traceback
from pathlib import Path
from loguru import logger

import RAG_config as rc

def RAG_QA_main():

    try:
        # Initialize configuration
        Baichuan_key = 'sk-eea01c3bf806eadc44c2c5b030e84907'
        config = rc.BaichuanConfig(
            api_key=Baichuan_key
        )

        # Initialize RAG system
        vector_store_base_path = "./vector_stores"
        rag_system = rc.RAGSystem(config, vector_store_base_path)

        # Document parameters
        pdf_path = "./方案.pdf" ### 患者日记.pdf
        vector_db_type = "FAISS"
        chunk_size = 800


        # Load or create vector store
        try:
            rag_system.load_or_create_vector_store(pdf_path, vector_db_type, chunk_size)
            print(f"\n知识库已加载: {pdf_path}")
        except Exception as e:
            logger.error(f"初始化向量存储失败: {str(e)}")
            raise

        # Interactive question loop
        print("\n=== 智能问答系统 ===")
        print("输入 'exit' 退出系统")
        print("输入 'help' 查看帮助信息")
        print("输入 'history' 查看历史记录")
        print("================================")

        conversation_history = []
        query_count = 0

        while True:
            try:
                # Get user input
                user_input = input("\n请输入您的问题 (输入 'exit' 退出): ").strip()

                # Check for exit command
                if user_input.lower() == 'exit':
                    break

                # Check for help command
                if user_input.lower() == 'help':
                    print("\n可用命令：")
                    print("- exit: 退出系统")
                    print("- help: 显示帮助信息")
                    print("- history: 查看历史对话记录")
                    continue

                # Check for history command
                if user_input.lower() == 'history':
                    if not conversation_history:
                        print("\n暂无历史对话记录")
                    else:
                        print("\n=== 历史对话记录 ===")
                        for entry in conversation_history:
                            print(f"\n问题: {entry['question']}")
                            print(f"回答: {entry['answer']}")
                            print("-" * 50)
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Process question
                query_count += 1
                query_id = f"q{query_count}"

                print("\n正在处理您的问题...")
                result = rag_system.answer_multiple_round_question(user_input, query_id)
                print("test result", result)
                print("test token ", result['answer']['token_usage'])
                # Store in history
                # conversation_history.append({
                #     "id": query_id,
                #     "question": user_input,
                #     "answer": result["answer"],
                #     "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                # })
                conversation_history.append({
                "id": query_id,
                "question": user_input,
                "answer": result["answer"],
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "token_usage": result['answer']['token_usage'],

            })

                # Display results
                # print("\n回答:", result["answer"])
                # if "context" in result:
                #     print(f"\n找到相关内容: {len(result['context'])} 字符")
                #     print(f"响应时间: {result['time_taken']:.2f} 秒")
                print("\n回答:", result["answer"])
                print("\ntoken:", result['answer']['token_usage'])
                if "context" in result:
                    print(f"\n找到相关内容: {len(result['context'])} 字符")
                    print(f"响应时间: {result['time_taken']:.2f} 秒")
                    if result["token_usage"]:
                        print("\n令牌使用情况:")
                        print(f"- 提示令牌: {result['answer']['token_usage']['prompt_tokens']}")
                        print(f"- 完成令牌: {result['answer']['token_usage']['completion_tokens']}")
                        print(f"- 总令牌数: {result['answer']['token_usage']['total_tokens']}")

            except KeyboardInterrupt:
                print("\n\n用户中断会话")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                print(f"\n错误: 处理问题失败 - {str(e)}")
                continue

        # Save conversation history
        if conversation_history:
            try:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                output_dir = Path('results')
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"对话记录_{timestamp}.json"

                with open(output_file, "w", encoding='utf-8') as f:
                    json.dump({
                        "session_info": {
                            "开始时间": conversation_history[0]["timestamp"],
                            "结束时间": conversation_history[-1]["timestamp"],
                            "问题总数": len(conversation_history)
                        },
                        "对话记录": conversation_history
                    }, f, ensure_ascii=False, indent=2)

                print(f"\n对话记录已保存至: {output_file}")

            except Exception as e:
                logger.error(f"保存对话记录时出错: {str(e)}")
                print("\n保存对话记录失败")

    except Exception as e:
        logger.error(f"系统运行失败: {str(e)}\n{traceback.format_exc()}")
        print("运行失败，请查看日志文件了解详情")
        raise
    finally:
        print("\n感谢使用智能问答系统！")


if __name__ == "__main__":
    RAG_QA_main()