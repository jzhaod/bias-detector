#!/usr/bin/env python3
"""
Simple test version of the bias detection MCP server to debug issues.
"""

import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastmcp import FastMCP
    print("✅ FastMCP imported successfully", file=sys.stderr)
except ImportError as e:
    print(f"❌ FastMCP import failed: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize simple MCP server
mcp = FastMCP("Bias Detector Test")

@mcp.tool()
def test_connection() -> str:
    """Simple test tool to verify MCP server is working."""
    return "Bias Detector MCP Server is working!"

@mcp.tool() 
def detect_bias_simple(text: str) -> dict:
    """Simple bias detection for testing."""
    return {
        "text": text,
        "biases_detected": [
            {
                "bias_name": "Test Bias",
                "severity": 5,
                "evidence": [text[:50] + "..."],
                "description": "This is a test detection"
            }
        ],
        "status": "test_mode"
    }

def main():
    """Main entry point."""
    print("🚀 Starting Test Bias Detection MCP Server...", file=sys.stderr)
    print("📋 Available tools: test_connection, detect_bias_simple", file=sys.stderr)
    
    try:
        mcp.run()
    except Exception as e:
        print(f"❌ Server failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()