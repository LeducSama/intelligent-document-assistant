# 📚 Multi-Document Usage Guide

## How to Switch Between Documents

The Intelligent Document Assistant supports multiple documents and allows you to focus on specific knowledge bases.

### **Method 1: Using the `docs` command**

1. **Type `docs` in the chat**:
```
🔐 You: docs
```

2. **View available documents**:
```
📚 Document Management
==============================
📖 Available Documents:
   1. KeePass2-GS.pdf (14 chunks) - 🟢 ACTIVE
   2. python_guide.txt (4 chunks) - ⚪ inactive
   3. api_reference.md (8 chunks) - ⚪ inactive

🛠️ Options:
1. Switch active documents
2. Add new document
3. Remove document
4. Back to chat
```

3. **Choose option 1 to switch**:
```
Enter choice (1-4): 1

📖 Select documents to activate (comma-separated numbers):
   1. KeePass2-GS.pdf
   2. python_guide.txt  
   3. api_reference.md

Enter numbers (e.g., 1,3): 2
✅ Activated documents: python_guide.txt
🔄 Conversation context reset for new document focus
```

4. **Now all questions search only the selected document(s)**

### **Method 2: Multiple Active Documents**

You can also search across multiple documents simultaneously:

```
Enter numbers (e.g., 1,3): 1,2,3
✅ Activated documents: KeePass2-GS.pdf, python_guide.txt, api_reference.md
```

### **Method 3: Adding New Documents**

1. **Choose option 2 in docs menu**:
```
Enter choice (1-4): 2

📄 Add New Document
Enter document path: /path/to/your/document.pdf
✅ Added and activated document: document.pdf
```

## **What Happens When You Switch**

1. **🔄 Context Reset**: Conversation history and user profiling reset for the new document focus
2. **🎯 Focused Search**: Only searches in the selected document(s)
3. **📊 Updated Stats**: System stats show which documents are active
4. **🧠 Smart Routing**: Intent classification adapts to the new document domain

## **Use Cases**

- **📖 Software Documentation**: Switch between different product manuals
- **🎓 Educational Materials**: Focus on specific course modules or textbooks
- **📋 Legal/Medical**: Switch between different regulatory documents
- **🔧 Technical Support**: Different product knowledge bases
- **📚 Research**: Focus on specific research papers or domains

## **Example Workflow**

```bash
# Start with KeePass documentation
🔐 You: How do I create a database?
🤖 Assistant: [KeePass-specific answer]

# Switch to Python documentation  
🔐 You: docs
[Select python_guide.txt]

🔐 You: How do I create a variable?
🤖 Assistant: [Python-specific answer]

# Switch to multiple documents
🔐 You: docs
[Select both KeePass and Python docs]

🔐 You: How do I create something?
🤖 Assistant: [Clarifying question about KeePass databases vs Python variables]
```

## **Pro Tips**

- **Use `status` command** to see which documents are currently active
- **Conversation context resets** when switching documents for clean focus
- **Add documents at runtime** without restarting the system
- **Remove unused documents** to keep the system clean
- **Mix and match documents** for cross-domain queries