#!/usr/bin/env python3
"""
Intelligent Document Assistant - Advanced LLM + RAG System

A production-ready conversational AI system that combines Large Language Models 
with Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware 
responses from document knowledge bases.

Key Features:
- Smart decision tree approach (direct answers vs clarifying questions)
- Conversation intelligence with intent classification
- User profiling that adapts to expertise level
- Hybrid semantic + keyword search
- Multiple LLM provider support (Gemini, OpenAI, Claude, Mock)
- Semantic chunking with overlapping context windows

Author: Philippe LOUBOUNGA
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
# import numpy as np  # Removed dependency
from datetime import datetime
import os
import requests
import sys


class RoutingDecision(Enum):
    DIRECT_SEARCH = "direct_search"
    NEEDS_LLM_ENHANCEMENT = "needs_llm_enhancement"
    CONTEXTUAL_SEARCH = "contextual_search"
    GREETING = "greeting"


class ResponseType(Enum):
    DIRECT_ANSWER = "direct_answer"
    CLARIFICATION_NEEDED = "clarification_needed"
    NOT_FOUND = "not_found"
    ERROR = "error"


class UserIntent(Enum):
    INSTALLATION = "installation"
    CONFIGURATION = "configuration"
    USAGE = "usage"
    TROUBLESHOOTING = "troubleshooting"
    GENERAL_INFO = "general_info"
    FOLLOW_UP = "follow_up"


class ExpertiseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    embeddings: Optional[List[float]] = None
    
    
@dataclass
class RetrievalResult:
    chunks: List[DocumentChunk]
    similarity_scores: List[float]
    query: str
    confidence: float
    
    def single_clear_answer(self) -> bool:
        """Check if there's exactly one high-confidence chunk"""
        return len(self.chunks) == 1 and self.confidence > 0.8
    
    def multiple_related_procedures(self) -> bool:
        """Check if multiple chunks contain procedures"""
        procedure_chunks = [c for c in self.chunks if 'procedure' in c.metadata.get('type', '')]
        return len(procedure_chunks) > 1
    
    def has_conflicting_procedures(self) -> bool:
        """Check if procedures conflict with each other"""
        if len(self.chunks) < 2:
            return False
        # Simple heuristic: if chunks have very different content patterns
        return len(set(c.metadata.get('section', '') for c in self.chunks)) > 1


@dataclass
class LLMResponse:
    content: str
    response_type: ResponseType
    confidence: float
    sources: List[str]
    clarification_questions: List[str] = None


@dataclass
class UserProfile:
    expertise_level: ExpertiseLevel
    primary_intents: List[UserIntent]
    preferred_response_style: str  # detailed, concise, step_by_step
    topics_of_interest: List[str]
    conversation_count: int = 0
    
    
@dataclass  
class ConversationContext:
    recent_intents: List[UserIntent]
    topics_discussed: List[str]
    unresolved_questions: List[str]
    user_satisfaction_signals: List[str]  # follow_ups, thanks, complaints
    context_window: int = 5  # Remember last 5 interactions


class GeminiProvider:
    """Google Gemini integration for production use"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.api_calls = 0
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    def generate_response(self, prompt: str, context_chunks: List[DocumentChunk], query: str) -> str:
        """Generate response using Gemini API"""
        self.api_calls += 1
        
        if not self.api_key:
            return "Google API key not configured. Using fallback response."
        
        # Build context from chunks
        context = "\n\n".join([f"Section: {chunk.metadata.get('section', 'Unknown')}\n{chunk.content}" 
                              for chunk in context_chunks])
        
        user_prompt = f"""You are a helpful assistant answering questions about KeePass password manager. Use ONLY the provided documentation context to answer.

Context from KeePass documentation:
{context}

User question: {query}

Rules:
- Only use information from the provided context above
- If the context doesn't contain the answer, say so clearly
- Be specific and actionable
- Include step-by-step instructions when relevant
- Keep responses concise but complete

Answer:"""

        try:
            response = requests.post(
                f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [{
                        "parts": [{
                            "text": user_prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return "No response generated by Gemini."
            else:
                return f"Gemini API Error ({response.status_code}): Using fallback response."
                
        except Exception as e:
            return "Error connecting to Gemini. Using fallback response."
    
    def enhance_query(self, query: str, conversation_context: List[str]) -> str:
        """Enhance vague queries using Gemini"""
        self.api_calls += 1
        
        if not self.api_key:
            return f"{query} KeePass password manager"
        
        context_str = "\n".join(conversation_context[-3:]) if conversation_context else "No previous context"
        
        prompt = f"""The user asked: "{query}"
Previous conversation context: {context_str}

This question is about KeePass password manager. If the query is vague or unclear, rephrase it to be more specific and searchable. If it's already clear, return it as-is.

Return only the enhanced query, nothing else:"""

        try:
            response = requests.post(
                f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 100
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    return f"{query} KeePass"
            else:
                return f"{query} KeePass"
                
        except Exception as e:
            return f"{query} KeePass password manager"
    
    def validate_response(self, response: str, source_chunks: List[DocumentChunk]) -> bool:
        """Validate response is grounded in sources"""
        self.api_calls += 1
        
        # Simple validation for now - check content overlap
        source_content = " ".join([chunk.content for chunk in source_chunks]).lower()
        response_lower = response.lower()
        
        response_words = set(response_lower.split())
        source_words = set(source_content.split())
        overlap = len(response_words.intersection(source_words))
        
        return overlap > 10
    
    def generate_clarification_questions(self, chunks: List[DocumentChunk], query: str) -> List[str]:
        """Generate clarification questions using Gemini"""
        self.api_calls += 1
        
        if not self.api_key:
            sections = [chunk.metadata.get('section', 'Unknown') for chunk in chunks]
            return [f"I found information about {', '.join(set(sections))}. Which specific area interests you?"]
        
        sections = [chunk.metadata.get('section', 'Unknown') for chunk in chunks]
        section_list = ', '.join(set(sections))
        
        prompt = f"""I found multiple relevant sections in KeePass documentation for the query "{query}":
{section_list}

Generate 2-3 specific clarification questions to help the user choose the most relevant information. Make the questions helpful and specific to KeePass usage.

Return only the questions, one per line:"""

        try:
            response = requests.post(
                f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                headers={
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 200
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    questions = [q.strip() for q in content.split('\n') if q.strip() and '?' in q]
                    return questions[:3] if questions else [f"Which of these areas are you interested in: {section_list}?"]
                else:
                    return [f"Which of these areas are you interested in: {section_list}?"]
            else:
                return [f"I found information about {section_list}. Which area interests you most?"]
                
        except Exception as e:
            return [f"Could you specify which aspect you're interested in: {section_list}?"]


class MockLLMProvider:
    """Mock LLM provider for development/testing without API costs"""
    
    def __init__(self):
        self.api_calls = 0
        
    def generate_response(self, prompt: str, context_chunks: List[DocumentChunk], 
                         query: str) -> str:
        """Generate response based on context chunks"""
        self.api_calls += 1
        
        if not context_chunks:
            return "I couldn't find specific information about your question in the available documentation."
        
        # Extract key information from chunks
        combined_content = "\n\n".join([chunk.content for chunk in context_chunks])
        
        # Simple response generation based on content
        if "install" in query.lower():
            if "system requirements" in combined_content.lower():
                return self._extract_installation_info(combined_content)
            
        elif "auto-type" in query.lower() or "autotype" in query.lower():
            if "auto-type" in combined_content.lower():
                return self._extract_autotype_info(combined_content)
                
        elif "database" in query.lower() and "create" in query.lower():
            if "database" in combined_content.lower():
                return self._extract_database_info(combined_content)
                
        elif "troubleshoot" in query.lower() or "problem" in query.lower() or "won't" in query.lower():
            if "troubleshoot" in combined_content.lower() or "issues" in combined_content.lower():
                return self._extract_troubleshooting_info(combined_content)
        
        # Generic response
        return f"Based on the KeePass documentation:\n\n{combined_content[:500]}..."
    
    def enhance_query(self, query: str, conversation_context: List[str]) -> str:
        """Enhance vague queries"""
        self.api_calls += 1
        
        # Simple query enhancement
        if len(query.split()) < 3:
            return f"{query} KeePass password manager setup configuration"
        
        return query
    
    def validate_response(self, response: str, source_chunks: List[DocumentChunk]) -> bool:
        """Validate if response is grounded in sources"""
        self.api_calls += 1
        
        # Simple validation: check if response contains content from sources
        source_content = " ".join([chunk.content for chunk in source_chunks]).lower()
        response_lower = response.lower()
        
        # Check for key overlapping terms
        response_words = set(response_lower.split())
        source_words = set(source_content.split())
        overlap = len(response_words.intersection(source_words))
        
        return overlap > 10  # Require reasonable overlap
    
    def generate_clarification_questions(self, chunks: List[DocumentChunk], query: str) -> List[str]:
        """Generate clarification questions when multiple procedures found"""
        self.api_calls += 1
        
        sections = [chunk.metadata.get('section', 'Unknown') for chunk in chunks]
        unique_sections = list(set(sections))
        
        if len(unique_sections) > 1:
            return [
                f"I found information about {' and '.join(unique_sections[:3])}. Which specific area are you interested in?",
                "Could you provide more details about your specific situation?",
                "What is your main goal with KeePass?"
            ]
        
        return [
            "Could you provide more details about what you're trying to accomplish?",
            "What specific outcome are you looking for?"
        ]
    
    def _extract_installation_info(self, content: str) -> str:
        """Extract installation-specific information"""
        lines = content.split('\n')
        install_section = []
        in_install_section = False
        
        for line in lines:
            if any(word in line.lower() for word in ['install', 'download', 'system requirements']):
                in_install_section = True
            elif line.strip() and line.startswith('#') and in_install_section:
                if 'install' not in line.lower():
                    break
            
            if in_install_section:
                install_section.append(line)
        
        if install_section:
            return "Here's how to install KeePass:\n\n" + '\n'.join(install_section[:20])
        else:
            return "To install KeePass, you'll need to download it from the official website and follow the installation wizard."
    
    def _extract_autotype_info(self, content: str) -> str:
        """Extract auto-type specific information"""
        lines = content.split('\n')
        autotype_section = []
        in_autotype_section = False
        
        for line in lines:
            if 'auto-type' in line.lower():
                in_autotype_section = True
            elif line.strip() and line.startswith('#') and in_autotype_section:
                if 'auto-type' not in line.lower():
                    break
            
            if in_autotype_section:
                autotype_section.append(line)
        
        if autotype_section:
            return "Here's how to use KeePass Auto-Type:\n\n" + '\n'.join(autotype_section[:15])
        else:
            return "Auto-Type allows KeePass to automatically enter your passwords. You can configure it in the entry properties and use Ctrl+Alt+A to activate it."
    
    def _extract_database_info(self, content: str) -> str:
        """Extract database creation information"""
        lines = content.split('\n')
        db_section = []
        in_db_section = False
        
        for line in lines:
            if any(word in line.lower() for word in ['database', 'create', 'new database']):
                in_db_section = True
            elif line.strip() and line.startswith('#') and in_db_section:
                if not any(word in line.lower() for word in ['database', 'create']):
                    break
            
            if in_db_section:
                db_section.append(line)
        
        if db_section:
            return "Here's how to create a new KeePass database:\n\n" + '\n'.join(db_section[:15])
        else:
            return "To create a new database: Go to File → New, choose a location, set a master password, and save your database file."
    
    def _extract_troubleshooting_info(self, content: str) -> str:
        """Extract troubleshooting information"""
        lines = content.split('\n')
        trouble_section = []
        in_trouble_section = False
        
        for line in lines:
            if any(word in line.lower() for word in ['troubleshoot', 'issues', 'problems', 'cannot', 'won\'t']):
                in_trouble_section = True
            elif line.strip() and line.startswith('#') and in_trouble_section:
                if not any(word in line.lower() for word in ['troubleshoot', 'issues', 'problems']):
                    break
            
            if in_trouble_section:
                trouble_section.append(line)
        
        if trouble_section:
            return "Here are troubleshooting steps:\n\n" + '\n'.join(trouble_section[:15])
        else:
            return "For troubleshooting: Check your master password, verify file permissions, and ensure .NET Framework is installed."


class SemanticEmbeddingProvider:
    """Advanced embedding provider using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model with graceful fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            # Don't auto-install in externally managed environments
            # User can manually install with: pip install sentence-transformers
            self.model = None
        except Exception:
            # Any other error, fallback gracefully
            self.model = None
                
    def build_vocabulary(self, documents: List[str]):
        """No vocabulary needed for sentence transformers"""
        if not self.model:
            # Fallback to TF-IDF if sentence-transformers failed
            self._fallback_to_tfidf(documents)
    
    def embed_text(self, text: str) -> List[float]:
        """Create semantic embedding for text"""
        if not self.model:
            return self._tfidf_embed(text)
            
        try:
            # Clean and prepare text
            clean_text = text.strip()
            if not clean_text:
                return [0.0] * 384  # Default embedding size
                
            # Generate semantic embedding
            embedding = self.model.encode(clean_text, convert_to_tensor=False)
            return embedding.tolist()
            
        except Exception as e:
            return self._tfidf_embed(text)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Ensure embeddings are same length
            if len(embedding1) != len(embedding2):
                return 0.0
                
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return max(0, similarity)  # Ensure non-negative
            
        except Exception as e:
            return 0.0
    
    # Fallback TF-IDF implementation
    def _fallback_to_tfidf(self, documents: List[str]):
        """Fallback to TF-IDF if sentence-transformers unavailable"""
        import re
        from collections import Counter
        
        self.embedding_dim = 384
        self.vocabulary = {}
        self.idf_scores = {}
        
        # Extract all words
        all_words = []
        for doc in documents:
            words = re.findall(r'\b\w+\b', doc.lower())
            all_words.extend(words)
        
        # Build vocabulary with most common words
        word_counts = Counter(all_words)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(self.embedding_dim))}
        
        # Calculate IDF scores
        import math
        for word in self.vocabulary:
            doc_count = sum(1 for doc in documents if word in doc.lower())
            self.idf_scores[word] = math.log(len(documents) / (doc_count + 1))
    
    def _tfidf_embed(self, text: str) -> List[float]:
        """TF-IDF embedding fallback"""
        if not hasattr(self, 'vocabulary'):
            return [0.0] * 384
            
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Create TF-IDF vector
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        embedding = [0.0] * self.embedding_dim
        for word, count in word_counts.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / len(words) if words else 0
                idf = self.idf_scores.get(word, 0)
                embedding[idx] = tf * idf
        
        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding


class LLMRAGSystem:
    """
    Main LLM + RAG System with Conversation Intelligence
    
    A production-ready system that combines semantic search, intent classification,
    user profiling, and a smart decision tree to provide contextual responses
    from document knowledge bases.
    
    Features:
    - Smart decision tree (direct answers vs clarifying questions)
    - Intent classification and user profiling
    - Hybrid semantic + keyword search
    - Conversation memory and context tracking
    - Multiple LLM provider support
    """
    
    def __init__(self, llm_provider=None, embedding_provider=None, use_gemini=False, gemini_api_key=None, gemini_model="gemini-1.5-flash"):
        if use_gemini and gemini_api_key:
            self.llm = GeminiProvider(api_key=gemini_api_key, model=gemini_model)
        elif llm_provider:
            self.llm = llm_provider
        else:
            self.llm = MockLLMProvider()
            
        self.embedder = embedding_provider or SemanticEmbeddingProvider()
        self.chunks: List[DocumentChunk] = []
        self.conversation_history: List[Dict[str, str]] = []
        
        # Multi-document support
        self.documents: Dict[str, str] = {}  # filename -> content
        self.active_documents: List[str] = []  # currently active document names
        self.document_chunks: Dict[str, List[DocumentChunk]] = {}  # filename -> chunks
        
        # Advanced conversation management
        self.user_profile = UserProfile(
            expertise_level=ExpertiseLevel.BEGINNER,
            primary_intents=[],
            preferred_response_style="step_by_step",
            topics_of_interest=[]
        )
        self.conversation_context = ConversationContext(
            recent_intents=[],
            topics_discussed=[],
            unresolved_questions=[],
            user_satisfaction_signals=[]
        )
        
    def load_documents(self, documents: Dict[str, str]):
        """Load documents and create embeddings"""
        self.documents.update(documents)
        
        # Process each document separately
        all_text = []
        for file_path, content in documents.items():
            chunks = self._split_into_chunks(content, file_path)
            self.document_chunks[file_path] = chunks
            all_text.extend([chunk.content for chunk in chunks])
        
        # Build vocabulary from all documents
        self.embedder.build_vocabulary(all_text)
        
        # Create embeddings for all chunks
        for file_path, chunks in self.document_chunks.items():
            for chunk in chunks:
                chunk.embeddings = self.embedder.embed_text(chunk.content)
        
        # Set all documents as active by default
        self.active_documents = list(documents.keys())
        self._update_active_chunks()
    
    def set_active_documents(self, document_names: List[str]):
        """Set which documents to search in"""
        available_docs = set(self.documents.keys())
        valid_docs = [doc for doc in document_names if doc in available_docs]
        
        if not valid_docs:
            raise ValueError(f"No valid documents found. Available: {list(available_docs)}")
        
        self.active_documents = valid_docs
        self._update_active_chunks()
        
        # Reset conversation context when switching documents
        self.conversation_context.recent_intents = []
        self.conversation_context.topics_discussed = []
        self.conversation_context.unresolved_questions = []
    
    def get_available_documents(self) -> List[str]:
        """Get list of all loaded document names"""
        return list(self.documents.keys())
    
    def get_active_documents(self) -> List[str]:
        """Get list of currently active document names"""
        return self.active_documents.copy()
    
    def _update_active_chunks(self):
        """Update the active chunks based on selected documents"""
        self.chunks = []
        for doc_name in self.active_documents:
            if doc_name in self.document_chunks:
                self.chunks.extend(self.document_chunks[doc_name])
    
    def add_document(self, file_path: str, content: str, set_active: bool = True):
        """Add a new document to the system"""
        # Process the new document
        chunks = self._split_into_chunks(content, file_path)
        
        # Create embeddings for new chunks
        for chunk in chunks:
            chunk.embeddings = self.embedder.embed_text(chunk.content)
        
        # Store the new document
        self.documents[file_path] = content
        self.document_chunks[file_path] = chunks
        
        # Optionally add to active documents
        if set_active:
            if file_path not in self.active_documents:
                self.active_documents.append(file_path)
                self._update_active_chunks()
    
    def remove_document(self, file_path: str):
        """Remove a document from the system"""
        if file_path in self.documents:
            del self.documents[file_path]
            del self.document_chunks[file_path]
            
            if file_path in self.active_documents:
                self.active_documents.remove(file_path)
                self._update_active_chunks()
                
            # Reset context if no documents remain active
            if not self.active_documents:
                self.conversation_context.recent_intents = []
                self.conversation_context.topics_discussed = []
                self.conversation_context.unresolved_questions = []
    
    def query(self, user_question: str, verbose: bool = False) -> LLMResponse:
        """Main query processing with decision tree approach and conversation intelligence"""
        if verbose:
            print(f"Processing query: {user_question}")
        
        # Step 1: Classify user intent and update profile
        intent = self._classify_intent(user_question)
        self._update_user_profile(user_question, intent)
        if verbose:
            print(f"Intent: {intent.value}, Expertise: {self.user_profile.expertise_level.value}")
        
        # Step 2: Route the query with context awareness
        routing = self._route_query_with_context(user_question, intent)
        if verbose:
            print(f"Routing decision: {routing.value}")
        
        # Step 3: Enhance query if needed (with conversation context)
        enhanced_query = user_question
        if routing == RoutingDecision.NEEDS_LLM_ENHANCEMENT:
            enhanced_query = self.llm.enhance_query(user_question, self._get_conversation_context())
            if verbose:
                print(f"Enhanced query: {enhanced_query}")
        
        # Step 4: RAG finds relevant sections (intent-aware)
        retrieval_result = self._retrieve_relevant_chunks_with_intent(enhanced_query, intent)
        if verbose:
            print(f"Found {len(retrieval_result.chunks)} relevant chunks (confidence: {retrieval_result.confidence:.2f})")
        
        # Step 5: Decision Tree Logic (expertise-aware)
        response = self._apply_decision_tree_with_context(retrieval_result, user_question, intent, verbose)
        
        # Step 6: Update conversation context and history
        self._update_conversation_context(user_question, intent, response)
        self.conversation_history.append({
            "question": user_question,
            "response": response.content,
            "type": response.response_type.value,
            "intent": intent.value,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _route_query(self, query: str) -> RoutingDecision:
        """Tier 1: Rule-based routing (no LLM overhead)"""
        query_lower = query.lower()
        
        # Handle greetings
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'help', 'hey']):
            return RoutingDecision.GREETING
        
        # Clear technical queries
        if any(keyword in query_lower for keyword in ['install', 'configure', 'setup', 'create', 'use', 'auto-type']):
            return RoutingDecision.DIRECT_SEARCH
        
        # Follow-up in conversation
        if len(self.conversation_history) > 0 and len(query.split()) < 5:
            return RoutingDecision.CONTEXTUAL_SEARCH
        
        # Vague or complex queries need enhancement
        if len(query.split()) < 3 or any(vague in query_lower for vague in ['broken', 'not working', 'problem', 'issue']):
            return RoutingDecision.NEEDS_LLM_ENHANCEMENT
        
        return RoutingDecision.DIRECT_SEARCH
    
    def _classify_intent(self, query: str) -> UserIntent:
        """Classify user intent from query"""
        query_lower = query.lower()
        
        # Installation intent
        if any(keyword in query_lower for keyword in ['install', 'download', 'setup', 'get started', 'requirement']):
            return UserIntent.INSTALLATION
        
        # Configuration intent  
        elif any(keyword in query_lower for keyword in ['configure', 'setting', 'option', 'preference', 'customize']):
            return UserIntent.CONFIGURATION
        
        # Troubleshooting intent
        elif any(keyword in query_lower for keyword in ['problem', 'issue', 'error', 'not working', 'broken', 'fix', 'help']):
            return UserIntent.TROUBLESHOOTING
        
        # Usage intent
        elif any(keyword in query_lower for keyword in ['how to', 'use', 'do', 'create', 'make', 'access', 'open']):
            return UserIntent.USAGE
        
        # Follow-up intent (short queries, pronouns)
        elif (len(query.split()) < 4 and 
              any(word in query_lower for word in ['what', 'how', 'why', 'where', 'it', 'that', 'this']) and
              len(self.conversation_history) > 0):
            return UserIntent.FOLLOW_UP
        
        # Default to general info
        return UserIntent.GENERAL_INFO
    
    def _update_user_profile(self, query: str, intent: UserIntent):
        """Update user profile based on query patterns"""
        self.user_profile.conversation_count += 1
        
        # Track intents
        self.user_profile.primary_intents.append(intent)
        if len(self.user_profile.primary_intents) > 10:
            self.user_profile.primary_intents.pop(0)  # Keep last 10
        
        # Infer expertise level from query complexity and vocabulary
        technical_terms = ['database', 'encryption', 'master key', 'plugin', 'auto-type', 'entry', 'group']
        advanced_terms = ['kdbx', 'xml', 'csv', 'trigger', 'field reference', 'spr', 'placeholders']
        
        query_lower = query.lower()
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        advanced_count = sum(1 for term in advanced_terms if term in query_lower)
        
        # Update expertise based on conversation patterns
        if self.user_profile.conversation_count > 5:
            if advanced_count > 0 or technical_count > 2:
                self.user_profile.expertise_level = ExpertiseLevel.ADVANCED
            elif technical_count > 0 or len(query.split()) > 8:
                self.user_profile.expertise_level = ExpertiseLevel.INTERMEDIATE
        
        # Extract topics of interest
        topics = []
        if 'auto-type' in query_lower:
            topics.append('auto-type')
        if any(word in query_lower for word in ['password', 'database']):
            topics.append('password_management')
        if any(word in query_lower for word in ['install', 'setup']):
            topics.append('installation')
        if any(word in query_lower for word in ['backup', 'sync']):
            topics.append('backup_sync')
            
        self.user_profile.topics_of_interest.extend(topics)
        # Keep unique topics, last 20
        self.user_profile.topics_of_interest = list(set(self.user_profile.topics_of_interest))[-20:]
    
    def _route_query_with_context(self, query: str, intent: UserIntent) -> RoutingDecision:
        """Enhanced routing with intent awareness"""
        query_lower = query.lower()
        
        # Handle greetings
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'help', 'hey']):
            return RoutingDecision.GREETING
        
        # Follow-up questions use conversation context
        if intent == UserIntent.FOLLOW_UP:
            return RoutingDecision.CONTEXTUAL_SEARCH
        
        # Clear technical queries with known intent
        if intent in [UserIntent.INSTALLATION, UserIntent.USAGE] and len(query.split()) > 3:
            return RoutingDecision.DIRECT_SEARCH
        
        # Troubleshooting usually needs enhancement unless very specific
        if intent == UserIntent.TROUBLESHOOTING:
            specific_indicators = ['error message', 'cannot', 'won\'t open', 'crashed']
            if any(indicator in query_lower for indicator in specific_indicators):
                return RoutingDecision.DIRECT_SEARCH
            else:
                return RoutingDecision.NEEDS_LLM_ENHANCEMENT
        
        # Vague or complex queries need enhancement
        if len(query.split()) < 3 or any(vague in query_lower for vague in ['broken', 'not working', 'problem', 'issue']):
            return RoutingDecision.NEEDS_LLM_ENHANCEMENT
        
        return RoutingDecision.DIRECT_SEARCH
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Hybrid retrieval using semantic + keyword matching"""
        # Get semantic scores
        semantic_scores = self._semantic_search(query)
        
        # Get keyword scores  
        keyword_scores = self._keyword_search(query)
        
        # Combine scores with weights
        hybrid_scores = []
        semantic_weight = 0.7  # Favor semantic search
        keyword_weight = 0.3
        
        for i, chunk in enumerate(self.chunks):
            semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.0
            keyword_score = keyword_scores[i] if i < len(keyword_scores) else 0.0
            
            # Hybrid score
            hybrid_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            hybrid_scores.append(hybrid_score)
        
        # Get top chunks
        chunk_scores = list(zip(self.chunks, hybrid_scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum threshold
        min_threshold = 0.05
        top_chunks = [chunk for chunk, score in chunk_scores[:top_k] if score > min_threshold]
        top_scores = [score for chunk, score in chunk_scores[:top_k] if score > min_threshold]
        
        # Calculate overall confidence
        confidence = max(top_scores) if top_scores else 0.0
        
        return RetrievalResult(
            chunks=top_chunks,
            similarity_scores=top_scores,
            query=query,
            confidence=confidence
        )
    
    def _semantic_search(self, query: str) -> List[float]:
        """Semantic search using embeddings"""
        query_embedding = self.embedder.embed_text(query)
        
        similarities = []
        for chunk in self.chunks:
            if chunk.embeddings:
                sim = self.embedder.similarity(query_embedding, chunk.embeddings)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        return similarities
    
    def _keyword_search(self, query: str) -> List[float]:
        """Keyword-based search with TF-IDF scoring"""
        import re
        from collections import Counter
        
        # Extract query terms
        query_terms = [term.lower() for term in re.findall(r'\b\w+\b', query)]
        if not query_terms:
            return [0.0] * len(self.chunks)
        
        # Calculate keyword scores for each chunk
        keyword_scores = []
        
        for chunk in self.chunks:
            chunk_text = chunk.content.lower()
            chunk_terms = re.findall(r'\b\w+\b', chunk_text)
            chunk_term_counts = Counter(chunk_terms)
            
            # Calculate score based on term frequency and matches
            score = 0.0
            total_terms = len(chunk_terms)
            
            for query_term in query_terms:
                if query_term in chunk_term_counts:
                    # TF score (term frequency)
                    tf = chunk_term_counts[query_term] / total_terms if total_terms > 0 else 0
                    
                    # Boost for exact matches
                    exact_match_boost = 1.5 if query_term in chunk_text else 1.0
                    
                    # Boost for important terms (section types)
                    importance_boost = 1.0
                    chunk_type = chunk.metadata.get('type', '').lower()
                    if any(term in query.lower() for term in ['install', 'setup', 'create']) and chunk_type == 'procedure':
                        importance_boost = 1.3
                    elif any(term in query.lower() for term in ['problem', 'error', 'fix']) and chunk_type == 'troubleshooting':
                        importance_boost = 1.3
                    
                    score += tf * exact_match_boost * importance_boost
            
            # Normalize by query length
            normalized_score = score / len(query_terms) if query_terms else 0.0
            keyword_scores.append(min(normalized_score, 1.0))  # Cap at 1.0
        
        return keyword_scores
    
    def _retrieve_relevant_chunks_with_intent(self, query: str, intent: UserIntent, top_k: int = 5) -> RetrievalResult:
        """Intent-aware retrieval with boosting for relevant section types"""
        # Get standard hybrid scores
        retrieval_result = self._retrieve_relevant_chunks(query, top_k)
        
        # Apply intent-based boosting
        boosted_chunks = []
        boosted_scores = []
        
        for chunk, score in zip(retrieval_result.chunks, retrieval_result.similarity_scores):
            chunk_type = chunk.metadata.get('type', '').lower()
            boost_factor = 1.0
            
            # Boost relevant sections based on intent
            if intent == UserIntent.INSTALLATION and chunk_type == 'procedure':
                boost_factor = 1.3
            elif intent == UserIntent.TROUBLESHOOTING and chunk_type == 'troubleshooting':
                boost_factor = 1.4
            elif intent == UserIntent.CONFIGURATION and chunk_type in ['procedure', 'reference']:
                boost_factor = 1.2
            elif intent == UserIntent.USAGE and chunk_type in ['procedure', 'informational']:
                boost_factor = 1.1
            
            boosted_score = min(score * boost_factor, 1.0)  # Cap at 1.0
            boosted_chunks.append(chunk)
            boosted_scores.append(boosted_score)
        
        # Re-sort by boosted scores
        chunk_score_pairs = list(zip(boosted_chunks, boosted_scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        final_chunks = [chunk for chunk, score in chunk_score_pairs]
        final_scores = [score for chunk, score in chunk_score_pairs]
        
        return RetrievalResult(
            chunks=final_chunks,
            similarity_scores=final_scores,
            query=query,
            confidence=max(final_scores) if final_scores else 0.0
        )
    
    def _apply_decision_tree_with_context(self, retrieval_result: RetrievalResult, original_query: str, intent: UserIntent, verbose: bool = False) -> LLMResponse:
        """Enhanced decision tree with user expertise and intent awareness"""
        
        # Adjust confidence thresholds based on user expertise
        if self.user_profile.expertise_level == ExpertiseLevel.ADVANCED:
            single_answer_threshold = 0.6  # Lower threshold for advanced users
            multiple_procedures_threshold = 0.4
        elif self.user_profile.expertise_level == ExpertiseLevel.INTERMEDIATE:
            single_answer_threshold = 0.7
            multiple_procedures_threshold = 0.5
        else:  # Beginner
            single_answer_threshold = 0.8  # Higher threshold for beginners
            multiple_procedures_threshold = 0.6
        
        # Decision Point 1: Is there exactly ONE clear procedure?
        if (retrieval_result.confidence > single_answer_threshold and 
            len(retrieval_result.chunks) == 1) or retrieval_result.single_clear_answer():
            
            if verbose:
                print("Single clear answer found - generating direct response")
            
            response_content = self._generate_expertise_aware_response(
                retrieval_result.chunks, original_query, intent
            )
            
            return LLMResponse(
                content=response_content,
                response_type=ResponseType.DIRECT_ANSWER,
                confidence=retrieval_result.confidence,
                sources=[chunk.metadata.get('source', 'Unknown') for chunk in retrieval_result.chunks]
            )
        
        # Decision Point 2: Are there multiple procedures?
        elif (retrieval_result.confidence > multiple_procedures_threshold and 
              len(retrieval_result.chunks) > 1) or retrieval_result.multiple_related_procedures():
            
            if verbose:
                print("Multiple procedures found - generating clarifying questions")
            
            clarification_questions = self.llm.generate_clarification_questions(
                retrieval_result.chunks, original_query
            )
            
            # Provide overview of options
            sections = [chunk.metadata.get('section', 'procedure') for chunk in retrieval_result.chunks]
            overview = f"I found multiple relevant procedures: {', '.join(set(sections))}. "
            
            return LLMResponse(
                content=overview + clarification_questions[0],
                response_type=ResponseType.CLARIFICATION_NEEDED,
                confidence=retrieval_result.confidence,
                sources=[],
                clarification_questions=clarification_questions
            )
        
        # Decision Point 3: No clear procedure found
        else:
            if verbose:
                print("No clear answer - asking for more context")
            
            if retrieval_result.chunks:
                # Some relevant content found - provide what we have
                response_content = self._generate_expertise_aware_response(
                    retrieval_result.chunks, original_query, intent
                )
                
                # Only ask for clarification if the response seems incomplete
                if len(response_content) < 100 or "unclear" in response_content.lower():
                    response_content += "\n\nCould you provide more specific details about what you're trying to accomplish?"
                    response_type = ResponseType.CLARIFICATION_NEEDED
                else:
                    response_type = ResponseType.DIRECT_ANSWER
                
                return LLMResponse(
                    content=response_content,
                    response_type=response_type,
                    confidence=retrieval_result.confidence,
                    sources=[chunk.metadata.get('source', 'Unknown') for chunk in retrieval_result.chunks]
                )
            else:
                # No relevant content found
                return LLMResponse(
                    content="I couldn't find specific information about your question in the KeePass documentation. Could you rephrase your question or provide more details about what you're trying to do?",
                    response_type=ResponseType.NOT_FOUND,
                    confidence=0.0,
                    sources=[]
                )
    
    def _generate_expertise_aware_response(self, chunks: List[DocumentChunk], query: str, intent: UserIntent, partial: bool = False) -> str:
        """Generate response tailored to user expertise level"""
        
        # Build context prompt based on expertise level
        if self.user_profile.expertise_level == ExpertiseLevel.BEGINNER:
            style_prompt = "Provide step-by-step instructions with explanations. Use simple language and avoid technical jargon."
        elif self.user_profile.expertise_level == ExpertiseLevel.INTERMEDIATE:
            style_prompt = "Provide clear instructions with some technical details. Include relevant context and tips."
        else:  # Advanced
            style_prompt = "Provide concise, technical instructions. Include advanced options and assume familiarity with KeePass."
        
        # Add intent-specific guidance
        if intent == UserIntent.TROUBLESHOOTING:
            style_prompt += " Focus on diagnosis and solution steps."
        elif intent == UserIntent.INSTALLATION:
            style_prompt += " Include system requirements and verification steps."
        
        if partial:
            style_prompt += " Provide what information is available and ask for clarification on missing details."
        
        return self.llm.generate_response(
            prompt=style_prompt,
            context_chunks=chunks,
            query=query
        )
    
    def _update_conversation_context(self, query: str, intent: UserIntent, response: LLMResponse):
        """Update conversation context with latest interaction"""
        
        # Update recent intents
        self.conversation_context.recent_intents.append(intent)
        if len(self.conversation_context.recent_intents) > self.conversation_context.context_window:
            self.conversation_context.recent_intents.pop(0)
        
        # Extract topics from query
        query_lower = query.lower()
        topics = []
        if 'auto-type' in query_lower:
            topics.append('auto-type')
        if any(word in query_lower for word in ['install', 'installation']):
            topics.append('installation')
        if any(word in query_lower for word in ['database', 'password']):
            topics.append('password_management')
        if any(word in query_lower for word in ['backup', 'sync']):
            topics.append('backup_sync')
        
        self.conversation_context.topics_discussed.extend(topics)
        if len(self.conversation_context.topics_discussed) > 20:
            self.conversation_context.topics_discussed = self.conversation_context.topics_discussed[-20:]
        
        # Track unresolved questions
        if response.response_type == ResponseType.CLARIFICATION_NEEDED:
            self.conversation_context.unresolved_questions.append(query)
            if len(self.conversation_context.unresolved_questions) > 5:
                self.conversation_context.unresolved_questions.pop(0)
        elif response.response_type == ResponseType.DIRECT_ANSWER:
            # Remove resolved questions on same topic
            self.conversation_context.unresolved_questions = [
                q for q in self.conversation_context.unresolved_questions 
                if not any(topic in q.lower() for topic in topics)
            ]
    
    def _apply_decision_tree(self, retrieval_result: RetrievalResult, original_query: str, verbose: bool = False) -> LLMResponse:
        """
        Decision Tree Approach:
        1. Is there exactly ONE clear procedure? → Generate direct answer
        2. Are there multiple procedures? → Generate clarifying questions
        3. No clear procedure found? → Ask for more context
        """
        
        # Decision Point 1: Is there exactly ONE clear procedure?
        if retrieval_result.single_clear_answer():
            if verbose:
                print("Single clear answer found - generating direct response")
            response_content = self.llm.generate_response(
                prompt="Generate a helpful response based on this documentation",
                context_chunks=retrieval_result.chunks,
                query=original_query
            )
            
            # Validate response is grounded
            is_grounded = self.llm.validate_response(response_content, retrieval_result.chunks)
            
            return LLMResponse(
                content=response_content,
                response_type=ResponseType.DIRECT_ANSWER,
                confidence=retrieval_result.confidence,
                sources=[chunk.metadata.get('source', 'Unknown') for chunk in retrieval_result.chunks]
            )
        
        # Decision Point 2: Are there multiple procedures?
        elif retrieval_result.multiple_related_procedures():
            if verbose:
                print("Multiple procedures found - generating clarifying questions")
            clarification_questions = self.llm.generate_clarification_questions(
                retrieval_result.chunks, original_query
            )
            
            # Provide overview of options
            sections = [chunk.metadata.get('section', 'procedure') for chunk in retrieval_result.chunks]
            overview = f"I found multiple relevant procedures: {', '.join(set(sections))}. "
            
            return LLMResponse(
                content=overview + clarification_questions[0],
                response_type=ResponseType.CLARIFICATION_NEEDED,
                confidence=retrieval_result.confidence,
                sources=[],
                clarification_questions=clarification_questions
            )
        
        # Decision Point 3: No clear procedure found
        else:
            if verbose:
                print("No clear answer - asking for more context")
            if retrieval_result.chunks:
                # Some relevant content found but not specific enough
                partial_content = self.llm.generate_response(
                    prompt="Provide partial information and ask for clarification",
                    context_chunks=retrieval_result.chunks,
                    query=original_query
                )
                return LLMResponse(
                    content=partial_content + "\n\nCould you provide more specific details about what you're trying to accomplish?",
                    response_type=ResponseType.CLARIFICATION_NEEDED,
                    confidence=retrieval_result.confidence,
                    sources=[chunk.metadata.get('source', 'Unknown') for chunk in retrieval_result.chunks]
                )
            else:
                # No relevant content found
                return LLMResponse(
                    content="I couldn't find specific information about your question in the KeePass documentation. Could you rephrase your question or provide more details about what you're trying to do?",
                    response_type=ResponseType.NOT_FOUND,
                    confidence=0.0,
                    sources=[]
                )
    
    def _split_into_chunks(self, content: str, source_file: str) -> List[DocumentChunk]:
        """Split document into semantic chunks with overlap"""
        chunks = []
        
        # First, split by sections (markdown headers) 
        sections = content.split('\n## ')
        
        for section_idx, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Clean up section
            if section_idx > 0:
                section = '## ' + section
            
            # Extract section title
            lines = section.split('\n')
            title = lines[0].replace('#', '').strip() if lines else f"Section {section_idx}"
            
            # For long sections, create semantic sub-chunks
            section_chunks = self._semantic_split_section(section, title, source_file, section_idx)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _semantic_split_section(self, section: str, title: str, source_file: str, section_idx: int) -> List[DocumentChunk]:
        """Split a section into semantic sub-chunks with overlap"""
        chunks = []
        
        # Determine section type
        section_type = self._classify_section_type(section)
        
        # If section is short, keep as single chunk
        if len(section) < 800:
            chunk_id = hashlib.md5(f"{source_file}_{section_idx}_{title}".encode()).hexdigest()[:12]
            chunks.append(DocumentChunk(
                content=section.strip(),
                metadata={
                    'source': source_file,
                    'section': title,
                    'type': section_type,
                    'index': section_idx,
                    'sub_chunk': 0
                },
                chunk_id=chunk_id
            ))
            return chunks
        
        # For longer sections, split by sentences with overlap
        sentences = self._split_into_sentences(section)
        
        # Group sentences into chunks (target ~400-600 words each)
        current_chunk = []
        current_length = 0
        chunk_num = 0
        target_chunk_size = 500  # words
        overlap_sentences = 2  # sentences to overlap
        
        for i, sentence in enumerate(sentences):
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed target size and we have content
            if current_length + sentence_words > target_chunk_size and current_chunk:
                # Create chunk
                chunk_content = ' '.join(current_chunk)
                chunk_id = hashlib.md5(f"{source_file}_{section_idx}_{chunk_num}_{title}".encode()).hexdigest()[:12]
                
                chunks.append(DocumentChunk(
                    content=chunk_content.strip(),
                    metadata={
                        'source': source_file,
                        'section': title,
                        'type': section_type,
                        'index': section_idx,
                        'sub_chunk': chunk_num
                    },
                    chunk_id=chunk_id
                ))
                
                # Start new chunk with overlap from previous chunk
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:] + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                chunk_num += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_words
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_id = hashlib.md5(f"{source_file}_{section_idx}_{chunk_num}_{title}".encode()).hexdigest()[:12]
            
            chunks.append(DocumentChunk(
                content=chunk_content.strip(),
                metadata={
                    'source': source_file,
                    'section': title,
                    'type': section_type,
                    'index': section_idx,
                    'sub_chunk': chunk_num
                },
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def _classify_section_type(self, content: str) -> str:
        """Classify section type based on content"""
        content_lower = content.lower()
        
        # Procedure indicators
        if any(word in content_lower for word in ['step', 'procedure', 'how to', 'install', 'setup', 'configure', 'create']):
            return 'procedure'
        
        # Troubleshooting indicators  
        elif any(word in content_lower for word in ['troubleshoot', 'problem', 'issue', 'error', 'fix', 'solve', 'debug']):
            return 'troubleshooting'
        
        # Reference/informational indicators
        elif any(word in content_lower for word in ['overview', 'introduction', 'about', 'description', 'features']):
            return 'reference'
        
        # FAQ indicators
        elif any(word in content_lower for word in ['question', 'faq', 'frequently', 'q:', 'a:']):
            return 'faq'
        
        return 'informational'
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving structure"""
        import re
        
        # Split on sentence boundaries while preserving structure
        # This regex handles: . ! ? followed by whitespace and capital letter or end
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|\d+\.|\n|$)', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _get_conversation_context(self) -> List[str]:
        """Get relevant conversation context"""
        return [msg["question"] for msg in self.conversation_history[-3:]]  # Last 3 exchanges
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics with conversation intelligence"""
        # Calculate intent distribution
        intent_counts = {}
        for intent in self.user_profile.primary_intents:
            intent_counts[intent.value] = intent_counts.get(intent.value, 0) + 1
        
        # Document statistics
        doc_stats = {}
        for doc_name in self.documents:
            chunks = self.document_chunks.get(doc_name, [])
            doc_stats[doc_name] = {
                'chunks': len(chunks),
                'active': doc_name in self.active_documents
            }
        
        return {
            'total_chunks': len(self.chunks),
            'conversation_turns': len(self.conversation_history),
            'llm_api_calls': getattr(self.llm, 'api_calls', 0),
            'avg_chunk_size': sum(len(chunk.content) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0,
            'user_expertise_level': self.user_profile.expertise_level.value,
            'primary_intents': intent_counts,
            'topics_of_interest': self.user_profile.topics_of_interest[-5:],  # Last 5
            'recent_intents': [intent.value for intent in self.conversation_context.recent_intents],
            'unresolved_questions': len(self.conversation_context.unresolved_questions),
            'total_documents': len(self.documents),
            'active_documents': self.active_documents.copy(),
            'document_stats': doc_stats
        }