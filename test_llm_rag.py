#!/usr/bin/env python3
"""
Test the proper LLM + RAG System with KeePass documentation
Demonstrates the decision tree approach with real vector embeddings and LLM integration
"""

from llm_rag_system import LLMRAGSystem, ResponseType
from pdf_processor import process_pdf_for_rag


def test_llm_rag_system():
    """Test the LLM+RAG system with KeePass PDF"""
    
    print(" Testing LLM + RAG System with Decision Tree Approach")
    print("=" * 60)
    
    # Initialize the system
    rag_system = LLMRAGSystem()
    
    # Load KeePass documentation
    print("\n Loading KeePass documentation...")
    pdf_result = process_pdf_for_rag("./KeePass2-GS.pdf")
    
    if not pdf_result['success']:
        print(f" Error loading PDF: {pdf_result['error']}")
        return
    
    # Load into RAG system
    documents = {
        "KeePass2-GS.pdf": pdf_result['content']
    }
    rag_system.load_documents(documents)
    
    print(f" System ready with {rag_system.get_stats()['total_chunks']} chunks")
    
    # Test queries following the decision tree
    test_cases = [
        {
            "name": "Clear Single Procedure Query",
            "query": "How do I install KeePass on Windows?",
            "expected": ResponseType.DIRECT_ANSWER
        },
        {
            "name": "Specific Feature Query", 
            "query": "How do I use auto-type feature?",
            "expected": ResponseType.DIRECT_ANSWER
        },
        {
            "name": "Database Creation Query",
            "query": "How do I create a new password database?",
            "expected": ResponseType.DIRECT_ANSWER
        },
        {
            "name": "Ambiguous Query (Multiple Procedures)",
            "query": "How do I set up KeePass?",
            "expected": ResponseType.CLARIFICATION_NEEDED
        },
        {
            "name": "Vague Troubleshooting Query",
            "query": "KeePass is not working",
            "expected": ResponseType.CLARIFICATION_NEEDED
        },
        {
            "name": "Unrelated Query",
            "query": "How do I bake a cake?",
            "expected": ResponseType.NOT_FOUND
        }
    ]
    
    print(f"\n Running {len(test_cases)} test cases...")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n Test {i}: {test_case['name']}")
        print(f"Query: \"{test_case['query']}\"")
        print("-" * 50)
        
        # Process query through RAG system
        response = rag_system.query(test_case['query'], verbose=True)
        
        # Display result
        response_icons = {
            ResponseType.DIRECT_ANSWER: "",
            ResponseType.CLARIFICATION_NEEDED: "", 
            ResponseType.NOT_FOUND: "",
            ResponseType.ERROR: ""
        }
        
        icon = response_icons.get(response.response_type, "")
        print(f"\n{icon} Response Type: {response.response_type.value}")
        print(f" Confidence: {response.confidence:.2f}")
        
        if response.sources:
            print(f" Sources: {', '.join(response.sources)}")
        
        print(f" Response:")
        print(response.content[:300] + "..." if len(response.content) > 300 else response.content)
        
        # Check if response type matches expectation
        success = response.response_type == test_case['expected']
        status = " PASS" if success else " FAIL"
        print(f"\n{status} (Expected: {test_case['expected'].value})")
        
        results.append({
            'test': test_case['name'],
            'query': test_case['query'],
            'expected': test_case['expected'].value,
            'actual': response.response_type.value,
            'confidence': response.confidence,
            'success': success
        })
    
    # Summary
    print(f"\n Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    # System stats
    stats = rag_system.get_stats()
    print(f"\n System Statistics:")
    print(f"   • Total Chunks: {stats['total_chunks']}")
    print(f"   • Conversation Turns: {stats['conversation_turns']}")
    print(f"   • LLM API Calls: {stats['llm_api_calls']}")
    print(f"   • Avg Chunk Size: {stats['avg_chunk_size']:.0f} chars")
    
    # Detailed results
    print(f"\n Detailed Results:")
    for result in results:
        status_icon = "" if result['success'] else ""
        print(f"{status_icon} {result['test']}: {result['actual']} (conf: {result['confidence']:.2f})")
    
    return rag_system, results


def interactive_demo():
    """Interactive demo of the LLM+RAG system"""
    print(f"\n Interactive Demo")
    print("=" * 30)
    print("Ask questions about KeePass! Type 'quit' to exit.")
    
    # Load system
    rag_system = LLMRAGSystem()
    pdf_result = process_pdf_for_rag("./KeePass2-GS.pdf")
    documents = {"KeePass2-GS.pdf": pdf_result['content']}
    rag_system.load_documents(documents)
    
    print(f" System ready! {rag_system.get_stats()['total_chunks']} chunks loaded")
    
    while True:
        try:
            question = input(f"\n Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print(" Goodbye!")
                break
            
            if not question:
                continue
            
            # Process question
            response = rag_system.query(question, verbose=False)
            
            # Display response
            response_icons = {
                ResponseType.DIRECT_ANSWER: "",
                ResponseType.CLARIFICATION_NEEDED: "",
                ResponseType.NOT_FOUND: "", 
                ResponseType.ERROR: ""
            }
            
            icon = response_icons.get(response.response_type, "")
            print(f"\n{icon} Assistant (confidence: {response.confidence:.2f}):")
            print(response.content)
            
            if response.clarification_questions:
                print(f"\n Follow-up questions you could ask:")
                for q in response.clarification_questions[1:3]:  # Show 2 more options
                    print(f"   • {q}")
            
        except KeyboardInterrupt:
            print(f"\n Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")


if __name__ == "__main__":
    # Run automated tests
    system, test_results = test_llm_rag_system()
    
    # Ask if user wants interactive demo
    print(f"\n" + "=" * 60)
    demo_choice = input("Would you like to try the interactive demo? (y/n): ").strip().lower()
    
    if demo_choice in ['y', 'yes']:
        interactive_demo()
    
    print(f"\n LLM + RAG System test completed!")
    print(f"This system properly implements:")
    print(f"    Vector embeddings for semantic search")
    print(f"    LLM integration for response generation")
    print(f"    Decision tree approach (1 procedure → direct answer)")
    print(f"    Smart clarification logic (multiple procedures → clarify)")
    print(f"    Grounded responses with source attribution")