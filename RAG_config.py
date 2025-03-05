from langchain_openai import OpenAIEmbeddings
import getpass
import os
import json
import time
import shutil
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import traceback

import requests
from loguru import logger
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# API Keys setup
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

openai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3-large` 
)
Baichuan_key = 'sk-eea01c3bf806eadc44c2c5b030e84907' 
qwen_key = 'sk-accc10c5863243aeb0cfb92f285a557d' 
# dashscope.api_key = qwen_key

# Configure logger
logger.add(
    "rag_system.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
    backtrace=True,
    diagnose=True
)
class VectorStoreManager:
    """Manages vector store operations with document-based storage"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "vector_store_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}

    def _save_metadata(self) -> bool:
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False

    def get_vector_store_path(self, file_path: str, chunk_size: int) -> Path:
        """Get path for vector store based only on document name and chunk size"""
        file_name = Path(file_path).stem
        # Create a deterministic store name without timestamp
        store_name = f"{file_name}_chunk{chunk_size}"
        return self.base_path / store_name

    def _get_store_metadata(self, store_name: str) -> Optional[Dict]:
        """Get metadata for a specific store"""
        return self.metadata.get(store_name)

    def should_recreate_store(self, file_path: str, chunk_size: int) -> bool:
        """Check if vector store should be recreated based on file changes"""
        store_path = self.get_vector_store_path(file_path, chunk_size)
        store_metadata = self._get_store_metadata(store_path.name)
        
        # If no store exists or no metadata, should recreate
        if not store_path.exists() or not store_metadata:
            return True
            
        try:
            # Check if source file has been modified
            source_file_path = Path(file_path)
            if not source_file_path.exists():
                return True
                
            # Get current file hash
            current_hash = self._get_file_hash(source_file_path)
            stored_hash = store_metadata.get('file_hash')
            
            # If hashes don't match or stored hash is missing, recreate
            if not stored_hash or current_hash != stored_hash:
                logger.info(f"Source file {file_path} has been modified, recreating vector store")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking store status: {e}")
            return True

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            raise

    def cleanup_old_stores(self, max_unused_days: int = 30):
        """Clean up stores that haven't been used recently"""
        try:
            current_time = time.time()
            max_age = max_unused_days * 24 * 60 * 60
            
            all_stores = [p for p in self.base_path.glob("*_chunk*") if p.is_dir()]
            
            for store_path in all_stores:
                try:
                    store_metadata = self._get_store_metadata(store_path.name)
                    if not store_metadata:
                        logger.warning(f"No metadata found for store: {store_path}")
                        continue
                    
                    last_used = time.mktime(time.strptime(store_metadata['last_used'], '%Y-%m-%d %H:%M:%S'))
                    if current_time - last_used > max_age:
                        shutil.rmtree(store_path)
                        logger.info(f"Removed unused vector store: {store_path}")
                        del self.metadata[store_path.name]
                            
                except Exception as e:
                    logger.error(f"Error processing store {store_path}: {e}")
                    continue
            
            self._save_metadata()
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def register_vector_store(self, file_path: str, chunk_size: int):
        """Register vector store metadata with file hash"""
        try:
            store_path = self.get_vector_store_path(file_path, chunk_size)
            file_hash = self._get_file_hash(Path(file_path))
            
            self.metadata[store_path.name] = {
                'file_path': str(file_path),
                'chunk_size': chunk_size,
                'file_hash': file_hash,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'last_used': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self._save_metadata()
        except Exception as e:
            logger.error(f"Error registering vector store: {e}")
            raise

    def update_last_used(self, file_path: str, chunk_size: int):
        """Update the last used timestamp for a store"""
        try:
            store_path = self.get_vector_store_path(file_path, chunk_size)
            if store_path.name in self.metadata:
                self.metadata[store_path.name]['last_used'] = time.strftime('%Y-%m-%d %H:%M:%S')
                self._save_metadata()
        except Exception as e:
            logger.error(f"Error updating last used timestamp: {e}")
@dataclass
class BaichuanConfig:
    """Configuration for Baichuan API"""
    api_key: str
    base_url: str = "https://api.baichuan-ai.com/v1"
    model: str = "Baichuan2-Turbo"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key cannot be empty")

class BaichuanChat:
    """Baichuan chat completion handler"""
    def __init__(self, config: BaichuanConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def generate_response(self, messages: List[Dict[str, str]], max_retries: int = 3) -> Dict[str, Any]:
        retry_delays = [1, 2, 4]
        last_error = None
        
        for attempt, delay in enumerate(retry_delays[:max_retries]):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                if not response or not response.choices:
                    raise ValueError("Invalid response format from API")
                    
                answer = response.choices[0].message.content
                if not answer or not answer.strip():
                    raise ValueError("Empty response received")
                
                # Return both the answer and token usage
                return {
                    "answer": answer.strip(),
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                    
        error_msg = f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise Exception(error_msg)

class RAGSystem:
    """RAG System implementation"""
    def __init__(self, config: BaichuanConfig, vector_store_base_path: str):
        self.config = config
        self.vector_store_manager = VectorStoreManager(vector_store_base_path)
        # Explicitly specify the model and dimensions
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536  # Specify the correct dimension
        )
        self.chat_model = BaichuanChat(config)
        self.db = None
        self.retriever = None
        logger.info("RAG System initialized successfully")

    def _process_pdf(self, pdf_path: str, chunk_size: int) -> List[Document]:
        """Process PDF file and split into chunks"""
        try:
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            logger.info(f"Loading PDF from {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            
            if not pages:
                raise ValueError("No content extracted from PDF")
            
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=50,
                separator="\n",
                keep_separator=True,
                strip_whitespace=True
            )
            
            docs = text_splitter.split_documents(pages)
            
            if not docs:
                raise ValueError("No documents created after splitting")
            
            logger.info(f"Created {len(docs)} document chunks")
            return docs
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _verify_vector_store(self):
        """Verify vector store functionality and dimensions"""
        try:
            # Get embedding dimension from a test query
            test_query = "测试查询"
            test_embedding = self.embeddings.embed_query(test_query)
            embedding_dim = len(test_embedding)
            
            # Verify FAISS index dimension
            if hasattr(self.db, 'index'):
                index_dim = self.db.index.d
                if index_dim != embedding_dim:
                    raise ValueError(f"Dimension mismatch: Index has {index_dim} dimensions, but embeddings have {embedding_dim} dimensions")
            
            # Test search
            test_results = self.db.similarity_search(test_query, k=1)
            if not test_results:
                raise ValueError("Vector store verification failed: no results returned")
                
            logger.info(f"Vector store verification successful. Dimension: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"Vector store verification failed: {str(e)}")
            raise

    def _setup_retriever(self):
        """Setup document retriever"""
        try:
            self.retriever = self.db.as_retriever(
                search_kwargs={"k": 3}
            )
            logger.info("Retriever setup completed")
        except Exception as e:
            logger.error(f"Failed to setup retriever: {str(e)}")
            raise

    def load_or_create_vector_store(self, pdf_path: str, vector_db_type: str, chunk_size: int) -> None:
        """Load existing vector store or create new one"""
        try:
            self.vector_store_manager.cleanup_old_stores()
            
            vector_store_path = self.vector_store_manager.get_vector_store_path(pdf_path, chunk_size)
            should_recreate = self.vector_store_manager.should_recreate_store(pdf_path, chunk_size)

            if should_recreate:
                logger.info(f"Creating new vector store at {vector_store_path}")
                if vector_store_path.exists():
                    shutil.rmtree(vector_store_path)

                docs = self._process_pdf(pdf_path, chunk_size)
                
                vector_db_type = vector_db_type.upper()
                if vector_db_type == "FAISS":
                    self.db = FAISS.from_documents(
                        docs, 
                        self.embeddings,
                        normalize_L2=True  # Add L2 normalization
                    )
                    self.db.save_local(str(vector_store_path))
                elif vector_db_type == "CHROMA":
                    self.db = Chroma.from_documents(
                        docs,
                        embedding=self.embeddings,
                        persist_directory=str(vector_store_path)
                    )
                    self.db.persist()
                else:
                    raise ValueError(f"Unsupported vector store type: {vector_db_type}")

                self._verify_vector_store()
                self.vector_store_manager.register_vector_store(pdf_path, chunk_size)
                
            else:
                logger.info(f"Loading existing vector store from {vector_store_path}")
                if vector_db_type.upper() == "FAISS":
                    self.db = FAISS.load_local(
                        str(vector_store_path),
                        self.embeddings,
                        normalize_L2=True,
                        allow_dangerous_deserialization=True
                    )
                else:
                    self.db = Chroma(
                        persist_directory=str(vector_store_path),
                        embedding_function=self.embeddings
                    )
                logger.info("Vector store loaded successfully")

            self._setup_retriever()
            
        except Exception as e:
            logger.error(f"Vector store operation failed: {str(e)}")
            raise



    def answer_question(self, question: str, query_id: str) -> Dict[str, Any]:
        # 单轮对话
        try:
            time_start = time.time()
            
            logger.debug(f"Starting retrieval for question: {question}")
            
            # embedding 信息
            question_embedding = self.embeddings.embed_query(question)
            embedding_dim = len(question_embedding)
            
            
            if hasattr(self.db, 'index'):
                index_dim = self.db.index.d
                if index_dim != embedding_dim:
                    raise ValueError(f"Dimension mismatch: Index has {index_dim} dimensions, but query has {embedding_dim} dimensions")
            
            relevant_docs = self.retriever.get_relevant_documents(question)
            logger.debug(f"Retrieved {len(relevant_docs)} documents")
            
            if not relevant_docs:
                logger.warning(f"No relevant documents found for question {query_id}")
                return {
                    "id": query_id,
                    "question": question,
                    "answer": "无法找到相关内容，请尝试重新表述问题。",
                    "context": "",
                    "time_taken": time.time() - time_start,
                    "token_usage": None
                }
            
            context = "\n".join([doc.page_content for doc in relevant_docs])
            logger.debug(f"Combined context length: {len(context)}")
            
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的临床研究人员，请根据提供的上下文准确回答问题。
                    问题写成json形式,"""
                },
                {
                    "role": "user",
                    "content": f"""基于以下内容：\n\n{context}\n\n问题：{question}\n\n请提供准确、完整的回答。
                                如果上下文中没有足够信息，请明确指出, 写成{{result:"no answer",reason:""}}。
                    如果问题需要推理，请解释你的推理过程，写在{{reason:""}}。"""
                }
            ]
            
            try:
                response = self.chat_model.generate_response(messages)
                logger.debug(f"Generated answer for question {query_id}")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise
            
            time_end = time.time()
            
            result = {
                "id": query_id,
                "question": question,
                "answer": response["answer"],
                "context": context,
                "context_length": len(context),
                "num_relevant_docs": len(relevant_docs),
                "time_taken": time_end - time_start,
                "token_usage": response["token_usage"]
            }
            
            logger.debug(f"Successfully processed question {query_id}")
            return result
            
        except Exception as e:
            error_msg = f"Error answering question {query_id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                "id": query_id,
                "question": question,
                "answer": f"处理问题时发生错误: {str(e)}",
                "error": error_msg,
                "time_taken": time.time() - time_start if 'time_start' in locals() else None,
                "token_usage": None
            }
        
    def answer_multiple_round_question(self, question: str, query_id: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        多轮对话回答问题
        
        Args:
            question: Current question
            query_id: Unique identifier for the query
            conversation_history: List of previous QA pairs in the conversation
        """
        try:
            time_start = time.time()
            
           
            question_embedding = self.embeddings.embed_query(question)
            embedding_dim = len(question_embedding)
            
            if hasattr(self.db, 'index'):
                index_dim = self.db.index.d
                if index_dim != embedding_dim:
                    raise ValueError(f"Dimension mismatch: Index has {index_dim} dimensions, but query has {embedding_dim} dimensions")
            
            # 回溯历史记录
            relevant_docs = self.retriever.get_relevant_documents(question)
            logger.debug(f"Retrieved {len(relevant_docs)} documents for question {query_id}")
            
            if not relevant_docs:
                logger.warning(f"No relevant documents found for question {query_id}")
                return {
                    "id": query_id,
                    "question": question,
                    "answer": "无法找到相关内容，请尝试重新表述问题。",
                    "context": "",
                    "time_taken": time.time() - time_start
                }
            
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # 准备对话历史
            messages = [
                {
                    "role": "system",
                    "content": """你是一个专业的临床研究人员，请根据提供的上下文和之前的对话历史准确回答问题。
                    请确保回答的连贯性，并在必要时参考之前的对话内容。
                    如果需要澄清或补充之前的回答，请明确指出。
                    问题写成json形式。
                    """
                }
            ]
            
            
            if conversation_history:
                messages.extend(conversation_history)
            
            # 增加当前问题
            messages.append({
                "role": "user",
                "content": f"""基于以下内容和之前的对话：

                            参考资料：
                            {context}

                            当前问题：{question}

                            请提供准确、完整的回答：
                            1. 如果上下文中没有足够信息，请说明并写成{{result:"no answer",reason:""}}
                            2. 如果需要结合之前对话内容，请在回答中说明
                            3. 如果需要推理，请解释推理过程，写在{{reason:""}}中"""
                                        })
            
            try:
                answer = self.chat_model.generate_response(messages)
            
                logger.debug(f"Generated answer for question {query_id}")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise
            
            time_end = time.time()
            
            result = {
                "id": query_id,
                "question": question,
                "answer": answer,
                "context": context,
                "context_length": len(context),
                "num_relevant_docs": len(relevant_docs),
                "time_taken": time_end - time_start
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error answering question {query_id}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                "id": query_id,
                "question": question,
                "answer": f"处理问题时发生错误: {str(e)}",
                "error": error_msg,
                "time_taken": time.time() - time_start if 'time_start' in locals() else None
            }

__all__ = ['BaichuanConfig', 'RAGSystem','VectorStoreManager']