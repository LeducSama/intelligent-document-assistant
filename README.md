# 🧠 Intelligent Document Assistant

An advanced **LLM + RAG** system with conversation intelligence that provides contextual, adaptive responses from document knowledge bases.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Innovation

Instead of overwhelming users with generic responses, this system uses **conversation intelligence** to provide exactly what users need:

## 🤖 Smart Decision Tree

```
User Query → RAG Retrieval → Decision Tree:
├─ Exactly ONE clear procedure? → Generate direct answer
├─ Multiple procedures found?   → Ask clarifying questions to pick the right one  
└─ No clear procedure found?    → Ask for more context about their situation
```

## 🏗️ Architecture

### LLM Integration Points
- **Response Generation**: Always uses LLM for natural language responses
- **Query Enhancement**: LLM enhances vague queries ("it's broken" → "KeePass password database not opening")
- **Clarification Questions**: LLM generates specific questions when multiple procedures found
- **Answer Validation**: LLM validates responses are grounded in source material

### RAG Components
- **Vector Embeddings**: TF-IDF based semantic search (easily replaceable with real embeddings)
- **Semantic Retrieval**: Finds relevant documentation chunks based on query similarity
- **Source Attribution**: Tracks which documents provided information
- **Confidence Scoring**: Evaluates how well retrieved content matches the query

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/LeducSama/intelligent-document-assistant.git
cd intelligent-document-assistant
pip install -r requirements.txt
```

### 2. Setup API Keys (Optional)
```bash
cp .env.example .env
# Edit .env and add your Google API key for Gemini
```

### 3. Run the Assistant
```bash
python3 chat.py
```

### 4. Multi-Document Usage
- Use `docs` command to manage documents
- Switch between knowledge bases in real-time
- Add PDFs and text files on the fly

### 5. Run Tests
```bash
python3 test_llm_rag.py
```
Comprehensive test suite demonstrating the decision tree approach.

## 💬 Chat Features

### Commands
- `help` - Show available commands and tips
- `status` - Display system status and configuration  
- `stats` - Show conversation statistics
- `verbose` - Toggle decision tree debugging
- `clear` - Clear conversation history
- `quit` - Exit

### LLM Provider Options
1. **Mock LLM** - Free, no API key needed, good responses based on content matching
2. **Google Gemini** - Free API with generous limits, requires `GOOGLE_API_KEY` environment variable

## 📁 Project Structure

```
├── llm_rag_system.py    # Core LLM+RAG system with decision tree
├── chat.py              # Terminal chat interface  
├── test_llm_rag.py      # Test suite
├── pdf_processor.py     # PDF text extraction
├── KeePass2-GS.pdf      # Knowledge base document
└── README.md            # This file
```

## 🧪 Decision Tree Examples

### Scenario 1: Clear Single Procedure
```
User: "How do I install KeePass?"
→ RAG finds single installation section
→ Decision: ONE clear procedure found
→ Response: Direct step-by-step installation guide
```

### Scenario 2: Multiple Procedures  
```
User: "How do I set up KeePass?"
→ RAG finds: installation, database creation, configuration sections
→ Decision: MULTIPLE procedures found
→ Response: "I found procedures for installation, database creation, and configuration. Which specific area interests you?"
```

### Scenario 3: Need More Context
```
User: "KeePass isn't working"
→ RAG finds troubleshooting content but query too vague
→ Decision: NO clear procedure (ambiguous)
→ Response: "I found troubleshooting information. Could you describe the specific problem you're experiencing?"
```

## 🔧 Technical Details

### Vector Embeddings
- Uses TF-IDF approach for semantic similarity
- Easily replaceable with sentence transformers, OpenAI embeddings, etc.
- Vocabulary built from document corpus

### LLM Providers
- **MockLLMProvider**: Pattern-based responses, no API costs
- **GeminiProvider**: Google Gemini integration with Flash/Pro model options
- Extensible architecture for additional LLM providers

### Smart Routing
- Rule-based routing avoids unnecessary LLM calls
- Direct search for clear technical queries
- LLM enhancement only for vague queries
- Contextual search for follow-up questions

## 📊 Example Output

```
🔐 You: How do I use auto-type?

🤖 Assistant: 📋 (confidence: 🟢 0.85)

Here's how to use KeePass Auto-Type:

### Auto-Type Feature
1. **Setup Auto-Type**: Configure in entry properties
2. **Use Auto-Type**: Select entry and press Ctrl+Alt+A  
3. **Custom Sequences**: Define specific key combinations
4. **Global Auto-Type**: Use Ctrl+Alt+A from any window

The Auto-Type feature allows KeePass to automatically enter your passwords 
into forms and applications, saving you from manual copy-paste operations.

📚 Source: KeePass2-GS.pdf
```

## 🎯 Key Benefits

1. **Strategic LLM Use**: Uses AI where it adds value, rules where they work better
2. **Progressive Narrowing**: Guides users to exact information through clarification
3. **Source Grounding**: Responses always tied to actual documentation
4. **Cost Efficient**: Minimal LLM calls through smart routing
5. **Extensible**: Easy to add new LLM providers or embedding methods

## 🔄 Conversation Flow

The system maintains conversation context and uses it for:
- Query enhancement based on previous questions
- Contextual search for follow-up questions  
- Progressive clarification to narrow down exact needs
- Avoiding repeated clarification on the same topics

This creates a natural, helpful experience where the assistant guides users to the exact information they need rather than overwhelming them with generic responses.