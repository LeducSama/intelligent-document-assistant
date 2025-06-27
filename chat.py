#!/usr/bin/env python3
"""
Terminal Chat Interface for Intelligent Document Assistant

Interactive terminal interface for the LLM + RAG system with conversation
intelligence, intent classification, and user profiling.
"""

import os
import sys
from llm_rag_system import LLMRAGSystem, ResponseType
from pdf_processor import process_pdf_for_rag

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
# Load .env on import
load_env_file()


class TerminalChat:
    """Interactive terminal chat interface"""
    
    def __init__(self):
        self.rag_system = None
        self.setup_complete = False
        
    def setup_system(self):
        """Setup the RAG system with user configuration"""
        print("KeePass Assistant - LLM + RAG System")
        print("=" * 50)
        
        # Ask about LLM provider
        print("\nChoose LLM Provider:")
        print("1. Mock LLM (Free, no API key needed)")
        print("2. Google Gemini (Free API key)")
        
        while True:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Please enter 1 or 2")
        
        use_gemini = choice == '2'
        gemini_key = None
        gemini_model = "gemini-1.5-flash"
        
        if use_gemini:
            # Get Google API key
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                gemini_key = input("Enter your Google API key: ").strip()
                if not gemini_key:
                    print("No API key provided. Switching to Mock LLM.")
                    use_gemini = False
                else:
                    # Ask for model choice
                    print("\nChoose Gemini model:")
                    print("1. gemini-1.5-flash (fastest, free)")
                    print("2. gemini-1.5-pro (most capable)")
                    
                    model_choice = input("Enter choice (1 or 2) [default: 1]: ").strip() or "1"
                    models = {"1": "gemini-1.5-flash", "2": "gemini-1.5-pro"}
                    gemini_model = models.get(model_choice, "gemini-1.5-flash")
        
        # Initialize system
        print(f"\nInitializing RAG System...")
        self.rag_system = LLMRAGSystem(
            use_gemini=use_gemini,
            gemini_api_key=gemini_key,
            gemini_model=gemini_model
        )
        
        # Load KeePass documentation
        print("Loading KeePass documentation...")
        pdf_result = process_pdf_for_rag("./KeePass2-GS.pdf")
        
        if not pdf_result['success']:
            print(f"Error loading PDF: {pdf_result['error']}")
            return False
        
        documents = {"KeePass2-GS.pdf": pdf_result['content']}
        self.rag_system.load_documents(documents)
        
        stats = self.rag_system.get_stats()
        print(f"System ready! Loaded {stats['total_chunks']} knowledge chunks")
        
        self.setup_complete = True
        return True
    
    def show_help(self):
        """Show help information"""
        help_text = """
KeePass Assistant Commands:

CHAT:
   • Just type your question naturally
   • Examples: "How do I install KeePass?"
            "What is auto-type?"
            "How do I create a database?"

COMMANDS:
   • help     - Show this help
   • status   - Show system status  
   • stats    - Show conversation statistics
   • docs     - Manage documents (list, switch, add)
   • clear    - Clear conversation history
   • verbose  - Toggle verbose mode (shows decision tree steps)
   • quit     - Exit the chat

TIPS:
   • Be specific about what you want to do
   • The system uses a decision tree approach:
     - 1 clear procedure → Direct answer
     - Multiple procedures → Clarifying questions
     - No clear match → Ask for more context
   • Follow-up questions help narrow down to exact answers

DECISION TREE:
   User Question → RAG Retrieval → Decision Tree:
   ├─ Exactly ONE procedure? → Direct Answer
   ├─ Multiple procedures?   → Clarification Questions  
   └─ No clear match?        → Ask for More Context
"""
        print(help_text)
    
    def show_status(self):
        """Show system status"""
        if not self.setup_complete:
            print("System not initialized")
            return
        
        stats = self.rag_system.get_stats()
        
        # Determine LLM type
        if hasattr(self.rag_system.llm, 'base_url'):
            if "generativelanguage.googleapis.com" in self.rag_system.llm.base_url:
                llm_type = f"Google Gemini ({getattr(self.rag_system.llm, 'model', 'unknown')})"
            else:
                llm_type = "Unknown API Provider"
        else:
            llm_type = "Mock LLM"
        
        print(f"\nSystem Status:")
        print(f"   LLM Provider: {llm_type}")
        print(f"   Knowledge Chunks: {stats['total_chunks']}")
        print(f"   Conversation Turns: {stats['conversation_turns']}")
        print(f"   LLM API Calls: {stats['llm_api_calls']}")
        print(f"   Avg Chunk Size: {stats['avg_chunk_size']:.0f} chars")
        print(f"   User Expertise: {stats['user_expertise_level'].title()}")
        if stats['topics_of_interest']:
            print(f"   Topics of Interest: {', '.join(stats['topics_of_interest'])}")
        if stats['recent_intents']:
            print(f"   Recent Intents: {', '.join(stats['recent_intents'][-3:])}")
        if stats['unresolved_questions'] > 0:
            print(f"   Unresolved Questions: {stats['unresolved_questions']}")
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Active Documents: {', '.join(stats['active_documents']) if stats['active_documents'] else 'None'}")
    
    def show_stats(self):
        """Show conversation statistics"""
        if not self.setup_complete:
            print("System not initialized")
            return
        
        history = self.rag_system.conversation_history
        if not history:
            print("No conversations yet")
            return
        
        # Analyze conversation types
        response_types = [turn['type'] for turn in history]
        type_counts = {}
        for rtype in response_types:
            type_counts[rtype] = type_counts.get(rtype, 0) + 1
        
        print(f"\nConversation Statistics:")
        print(f"   Total Questions: {len(history)}")
        for rtype, count in type_counts.items():
            emoji = {"direct_answer": "-", "clarification_needed": "-", "not_found": "-", "error": "-"}.get(rtype, "-")
            print(f"   {emoji} {rtype.replace('_', ' ').title()}: {count}")
        
        # Show recent questions
        print(f"\nRecent Questions:")
        for turn in history[-3:]:
            print(f"   Q: {turn['question'][:60]}{'...' if len(turn['question']) > 60 else ''}")
    
    def manage_documents(self):
        """Document management interface"""
        if not self.setup_complete:
            print("System not initialized")
            return
        
        while True:
            print(f"\nDocument Management")
            print("=" * 30)
            
            # Show current status
            available_docs = self.rag_system.get_available_documents()
            active_docs = self.rag_system.get_active_documents()
            
            print(f"Available Documents:")
            for i, doc in enumerate(available_docs, 1):
                status = "ACTIVE" if doc in active_docs else "inactive"
                chunks = len(self.rag_system.document_chunks.get(doc, []))
                print(f"   {i}. {doc} ({chunks} chunks) - {status}")
            
            print(f"\nOptions:")
            print("1. Switch active documents")
            print("2. Add new document")
            print("3. Remove document") 
            print("4. Back to chat")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                self._switch_documents(available_docs)
            elif choice == '2':
                self._add_document()
            elif choice == '3':
                self._remove_document(available_docs)
            elif choice == '4':
                break
            else:
                print("Please enter 1, 2, 3, or 4")
    
    def _switch_documents(self, available_docs):
        """Switch which documents are active"""
        if not available_docs:
            print("No documents available")
            return
        
        print(f"\nSelect documents to activate (comma-separated numbers):")
        for i, doc in enumerate(available_docs, 1):
            print(f"   {i}. {doc}")
        
        selection = input("\nEnter numbers (e.g., 1,3): ").strip()
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_docs = [available_docs[i] for i in indices if 0 <= i < len(available_docs)]
            
            if selected_docs:
                self.rag_system.set_active_documents(selected_docs)
                print(f"Activated documents: {', '.join(selected_docs)}")
                print("Conversation context reset for new document focus")
            else:
                print("No valid documents selected")
                
        except (ValueError, IndexError):
            print("Invalid selection format")
    
    def _add_document(self):
        """Add a new document to the system"""
        print(f"\nAdd New Document")
        file_path = input("Enter document path: ").strip()
        
        if not file_path:
            print("No file path provided")
            return
        
        try:
            from pdf_processor import process_pdf_for_rag
            
            if file_path.endswith('.pdf'):
                result = process_pdf_for_rag(file_path)
                if result['success']:
                    content = result['content']
                    doc_name = file_path.split('/')[-1]  # Get filename
                else:
                    print(f"Error processing PDF: {result['error']}")
                    return
            else:
                # Try to read as text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_name = file_path.split('/')[-1]
            
            # Add to system
            self.rag_system.add_document(doc_name, content, set_active=True)
            print(f"Added and activated document: {doc_name}")
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error adding document: {e}")
    
    def _remove_document(self, available_docs):
        """Remove a document from the system"""
        if not available_docs:
            print("No documents to remove")
            return
        
        print(f"\nSelect document to remove:")
        for i, doc in enumerate(available_docs, 1):
            print(f"   {i}. {doc}")
        
        try:
            choice = int(input("\nEnter number: ").strip()) - 1
            if 0 <= choice < len(available_docs):
                doc_to_remove = available_docs[choice]
                
                confirm = input(f" Really remove '{doc_to_remove}'? (y/N): ").strip().lower()
                if confirm == 'y':
                    self.rag_system.remove_document(doc_to_remove)
                    print(f"Removed document: {doc_to_remove}")
                else:
                    print("Cancelled")
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")
    
    def run(self):
        """Main chat loop"""
        if not self.setup_system():
            print("Failed to initialize system")
            return
        
        print(f"\nChat started! Type 'help' for commands, 'quit' to exit")
        print("=" * 50)
        
        verbose_mode = False
        
        try:
            while True:
                # Get user input
                user_input = input(f"\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                elif user_input.lower() == 'docs':
                    self.manage_documents()
                    continue
                elif user_input.lower() == 'clear':
                    self.rag_system.conversation_history = []
                    print("Conversation history cleared")
                    continue
                elif user_input.lower() == 'verbose':
                    verbose_mode = not verbose_mode
                    print(f"Verbose mode: {'ON' if verbose_mode else 'OFF'}")
                    continue
                
                # Process question through RAG system
                print("Assistant: ", end="", flush=True)
                response = self.rag_system.query(user_input, verbose=verbose_mode)
                
                # Display response with appropriate formatting
                self._display_response(response)
                
        except KeyboardInterrupt:
            print(f"\n\nChat interrupted. Goodbye!")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        
        # Show final stats
        print(f"\nFinal Statistics:")
        self.show_stats()
        print(f"\nThank you for using KeePass Assistant!")
    
    def _display_response(self, response):
        """Display response with proper formatting"""
        
        # Response type indicator
        type_indicators = {
            ResponseType.DIRECT_ANSWER: "-",
            ResponseType.CLARIFICATION_NEEDED: "-", 
            ResponseType.NOT_FOUND: "-",
            ResponseType.ERROR: "-"
        }
        
        indicator = type_indicators.get(response.response_type, "-")
        
        # Confidence indicator  
        if response.confidence >= 0.8:
            conf_color = "HIGH"
        elif response.confidence >= 0.5:
            conf_color = "MED"
        else:
            conf_color = "LOW"
        
        print(f"{indicator} (confidence: {conf_color} {response.confidence:.2f})\n")
        
        # Main response content
        print(response.content)
        
        # Additional information
        if response.sources and response.response_type == ResponseType.DIRECT_ANSWER:
            print(f"\nSource: {', '.join(response.sources)}")
        
        if response.clarification_questions and len(response.clarification_questions) > 1:
            print(f"\nYou could also ask:")
            for question in response.clarification_questions[1:3]:  # Show 2 more options
                print(f"   • {question}")
        
        # Show follow-up suggestions for certain response types
        if response.response_type == ResponseType.CLARIFICATION_NEEDED:
            print(f"\nTry being more specific or providing additional context.")
        elif response.response_type == ResponseType.NOT_FOUND:
            print(f"Try rephrasing your question or asking about a different KeePass topic.")


def main():
    """Main entry point"""
    try:
        chat = TerminalChat()
        chat.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()