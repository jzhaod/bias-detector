#!/usr/bin/env python3
"""
Setup NLTK data for the bias detection server.
"""

import sys

try:
    import nltk
    print("Downloading NLTK data...", file=sys.stderr)
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('vader_lexicon', quiet=True)
    
    print("✅ NLTK data downloaded successfully", file=sys.stderr)
    
except Exception as e:
    print(f"❌ NLTK setup failed: {e}", file=sys.stderr)
    sys.exit(1)