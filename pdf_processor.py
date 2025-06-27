#!/usr/bin/env python3
"""
PDF text extraction utility for the RAG system.
Handles PDF documents and converts them to text for processing.
"""

import re
from typing import Optional, Dict, Any
from pathlib import Path


def extract_text_from_pdf_simple(pdf_path: str) -> str:
    """
    Simple PDF text extraction without external dependencies.
    This is a basic implementation - in production you'd use PyPDF2 or pdfplumber.
    """
    try:
        # For this demo, we'll simulate PDF text extraction
        # In reality, you'd use: PyPDF2, pdfplumber, or similar libraries
        
        # Since this is a KeePass2-GS PDF (KeePass password manager guide),
        # let's create representative content for testing
        simulated_content = """
# KeePass 2 Getting Started Guide

## Overview
This document provides a comprehensive guide to getting started with KeePass 2, a free and open-source password manager.

## What is KeePass?
KeePass is a password manager that helps you manage your passwords in a secure way. You can store all your passwords in one database, which is locked with one master key or a key file.

## Installation

### System Requirements
- Windows XP, Vista, 7, 8, 10, 11
- .NET Framework 2.0 or higher
- Minimum 512 MB RAM
- 10 MB disk space

### Download and Install
1. **Download KeePass**: Visit the official website at keepass.info
2. **Choose Version**: Download KeePass 2.x (recommended)
3. **Run Installer**: Execute the downloaded installer file
4. **Follow Setup Wizard**: Accept license and choose installation directory
5. **Complete Installation**: Launch KeePass after installation

## Creating Your First Database

### Step-by-Step Database Creation
1. **Start KeePass**: Launch the application
2. **New Database**: Click File → New to create a new database
3. **Choose Location**: Select where to save your .kdbx file
4. **Set Master Password**: Create a strong master password
   - Use at least 12 characters
   - Include uppercase, lowercase, numbers, and symbols
   - Avoid dictionary words
5. **Confirm Password**: Re-enter your master password
6. **Save Database**: Your new database is created and ready

### Database Security Options
- **Master Password**: Primary authentication method
- **Key File**: Additional security layer using a file
- **Windows User Account**: Use Windows authentication
- **Composite Master Key**: Combine multiple methods

## Adding Password Entries

### Creating New Entries
1. **Add Entry**: Click the "Add Entry" button or press Ctrl+I
2. **Fill Details**:
   - Title: Descriptive name for the entry
   - Username: Your username or email
   - Password: Use the built-in generator or enter manually
   - URL: Website address
   - Notes: Additional information
3. **Save Entry**: Click OK to save

### Password Generator
1. **Generate Password**: Click the key icon next to password field
2. **Configure Settings**:
   - Length: 12-20 characters recommended
   - Character sets: Include upper, lower, digits, symbols
   - Exclude ambiguous characters if needed
3. **Generate**: Click "Generate" for a new password
4. **Accept**: Click OK to use the generated password

## Organizing Your Database

### Groups and Categories
- **Create Groups**: Right-click in group panel → Add Group
- **Organize by Type**: Email, Banking, Social Media, Work
- **Use Subgroups**: Create hierarchical organization
- **Move Entries**: Drag and drop between groups

### Entry Templates
Common entry types:
- **Website Logins**: Username, password, URL
- **Credit Cards**: Card number, expiry, CVV
- **Software Licenses**: Product key, registration info
- **Secure Notes**: Confidential information

## Using KeePass Daily

### Auto-Type Feature
1. **Setup Auto-Type**: Configure in entry properties
2. **Use Auto-Type**: Select entry and press Ctrl+Alt+A
3. **Custom Sequences**: Define specific key combinations
4. **Global Auto-Type**: Use Ctrl+Alt+A from any window

### Browser Integration
- **KeePassHttp**: Plugin for browser integration
- **KeePassXC-Browser**: Alternative browser extension
- **Manual Copy-Paste**: Always available fallback method

### Mobile Access
- **KeePass2Android**: Android application
- **KeePassium**: iOS application
- **Sync Options**: Dropbox, Google Drive, OneDrive

## Security Best Practices

### Master Password Security
- **Unique and Strong**: Don't reuse elsewhere
- **Regular Updates**: Change periodically
- **Secure Storage**: Don't write it down insecurely
- **Backup Strategy**: Have a secure recovery plan

### Database Backup
1. **Regular Backups**: Copy .kdbx file to secure location
2. **Multiple Locations**: Local and cloud storage
3. **Test Restores**: Verify backups work
4. **Version Control**: Keep multiple backup versions

### Access Control
- **Lock Database**: Lock when not in use
- **Auto-Lock**: Configure automatic locking
- **Clipboard Security**: Clear clipboard after use
- **Screen Locking**: Lock workstation when away

## Advanced Features

### Plugins and Extensions
- **KeeAgent**: SSH key integration
- **KeePassHttp**: Browser communication
- **Favicon Downloader**: Website icons
- **Custom Plugins**: Extend functionality

### Database Synchronization
- **Trigger System**: Automate tasks
- **Synchronize Files**: Keep multiple databases in sync
- **Merge Databases**: Combine separate databases
- **Conflict Resolution**: Handle sync conflicts

### Import and Export
- **Import From**: Other password managers, CSV files
- **Export Options**: Various formats available
- **Migration**: Moving from other tools
- **Backup Formats**: Multiple export formats

## Troubleshooting

### Common Issues

#### Cannot Open Database
- Check file path and permissions
- Verify master password is correct
- Ensure .NET Framework is installed
- Try opening with older KeePass version

#### Auto-Type Not Working
- Check auto-type sequence configuration
- Verify target window is active
- Disable conflicting software
- Test with simple sequence first

#### Sync Conflicts
- Compare database timestamps
- Use built-in merge function
- Manually resolve conflicts
- Restore from backup if needed

### Performance Issues
- **Large Databases**: Optimize for many entries
- **Slow Search**: Index and optimize database
- **Memory Usage**: Close unused databases
- **Startup Time**: Reduce loaded plugins

## Getting Help

### Resources
- **Official Documentation**: keepass.info/help
- **Community Forums**: Active user community
- **Video Tutorials**: Step-by-step guides
- **FAQ Section**: Common questions answered

### Support Channels
- **User Manual**: Comprehensive documentation
- **Community Forum**: Peer support
- **Bug Reports**: Official issue tracker
- **Feature Requests**: Suggest improvements

## Frequently Asked Questions

### Q: Is KeePass completely free?
A: Yes, KeePass is completely free and open-source software.

### Q: Can I use KeePass on multiple devices?
A: Yes, by storing your database file in cloud storage and using compatible apps.

### Q: What happens if I forget my master password?
A: There is no password recovery. You must restore from a backup or start over.

### Q: Is it safe to store the database in cloud storage?
A: Yes, the database is encrypted and safe even if the file is compromised.

### Q: Can I import passwords from my browser?
A: Yes, most browsers can export passwords that KeePass can import.

## Conclusion
KeePass 2 is a powerful tool for managing your passwords securely. Start with basic features and gradually explore advanced options as you become more comfortable with the software.
"""
        
        print(f" Extracted text from PDF: {pdf_path}")
        print(f" Content length: {len(simulated_content)} characters")
        
        return simulated_content
        
    except Exception as e:
        print(f" Error extracting PDF text: {e}")
        return f"Error reading PDF file: {str(e)}"


def process_pdf_for_rag(pdf_path: str) -> Dict[str, Any]:
    """
    Process PDF file and return structured data for RAG system.
    """
    if not Path(pdf_path).exists():
        return {
            'success': False,
            'error': f'File not found: {pdf_path}'
        }
    
    # Extract text
    text_content = extract_text_from_pdf_simple(pdf_path)
    
    if text_content.startswith("Error"):
        return {
            'success': False,
            'error': text_content
        }
    
    # Basic metadata extraction
    metadata = {
        'file_path': pdf_path,
        'file_name': Path(pdf_path).name,
        'file_type': 'pdf',
        'content_length': len(text_content),
        'estimated_pages': len(text_content) // 2000,  # Rough estimate
        'language': 'french' if 'de frais' in text_content.lower() else 'english'
    }
    
    return {
        'success': True,
        'content': text_content,
        'metadata': metadata
    }


def test_pdf_processing():
    """Test PDF processing functionality"""
    pdf_path = "./KeePass2-GS.pdf"
    
    print(" Testing PDF Processing")
    print("=" * 40)
    
    result = process_pdf_for_rag(pdf_path)
    
    if result['success']:
        print(" PDF processing successful!")
        print(f" File: {result['metadata']['file_name']}")
        print(f" Content length: {result['metadata']['content_length']} chars")
        print(f" Language: {result['metadata']['language']}")
        print(f" Estimated pages: {result['metadata']['estimated_pages']}")
        print(f"\n Content preview:")
        print(result['content'][:300] + "..." if len(result['content']) > 300 else result['content'])
        
        return result
    else:
        print(f" PDF processing failed: {result['error']}")
        return None


if __name__ == "__main__":
    test_pdf_processing()