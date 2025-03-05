import sys
import os
import time
import json
import traceback
from pathlib import Path
from loguru import logger

import RAG_config as rc

# Configure logger if not already configured in RAG_Config
logger.add(
    "icf_extraction.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
)

def generate_icf_questions(ICF_components):
    """Generate questions for ICF components"""
    ICF_questions = []
    for i, component in enumerate(ICF_components, 1):
        question = {
            "id": f"q{i}",
            "question": f"请问本方案的{component}是什么？"
        }
        ICF_questions.append(question)
        logger.info(f"Generated question {i}: {question['question']}")

    return ICF_questions  # Fixed variable name

def ICF_main():
    try:
        # Initialize configuration with proper error handling
        Baichuan_key = os.getenv("BAICHUAN_KEY", 'sk-eea01c3bf806eadc44c2c5b030e84907')
        config = rc.BaichuanConfig(
            api_key=Baichuan_key
        )

        # Initialize RAG system
        vector_store_base_path = "./vector_stores"
        os.makedirs(vector_store_base_path, exist_ok=True)  # Create directory if it doesn't exist
        rag_system = rc.RAGSystem(config, vector_store_base_path)

        # Document parameters
        pdf_path = "./方案.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        vector_db_type = "FAISS"
        chunk_size = 800

        # Load or create vector store
        try:
            rag_system.load_or_create_vector_store(pdf_path, vector_db_type, chunk_size)
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

        # Generate ICF questions
        ICF_components = ["方案名称", "方案伦理编号", "主要研究者", "研究目的", "研究过程"]
        questions = generate_icf_questions(ICF_components)

        # Process questions
        logger.info("Processing ICF questions...")
        results = []

        # Process initial questions
        for q in questions:
            try:
                logger.info(f"Processing question: {q['question']}")
                result = rag_system.answer_question(q["question"], q["id"])
                print(result)
                # print("token_usage",result['answer']['token_usage'])
                results.append(result)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                error_msg = f"Error processing question {q['id']}: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "answer": f"处理问题时发生错误: {str(e)}",
                    "error": str(e)
                })

        # Process re-identification questions
        re_identify_components = ["方案伦理编号"]
        re_identify_questions = generate_icf_questions(re_identify_components)
        for q in re_identify_questions:
            try:
                logger.info(f"Processing re-identify question: {q['question']}")
                result = rag_system.answer_question(q["question"], q["id"])
                results.append(result)
                time.sleep(0.5)
            except Exception as e:
                error_msg = f"Error processing question {q['id']}: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "answer": f"处理问题时发生错误: {str(e)}",
                    "error": str(e)
                })

        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"ICF_Results_Baichuan_{timestamp}.json"

        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Log summary statistics
        successful_queries = sum(1 for r in results if 'error' not in r)
        logger.info(f"Total questions processed: {len(results)}")
        logger.info(f"Successful queries: {successful_queries}")
        logger.info(f"Failed queries: {len(results) - successful_queries}")
        logger.info(f"Results saved to: {output_file}")

        return results, output_file

    except Exception as e:
        error_msg = f"Main execution failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    try:
        results, output_file = ICF_main()
        print(f"\nProcessing complete. Results saved to: {output_file}")
    except Exception as e:
        print(f"Execution failed: {str(e)}")
        print("Please check the log file for details.")