# Bias Detection MCP Server

A dual-interface MCP server for cognitive bias detection using Charlie Munger's 25 psychological tendencies framework. Provides both fast heuristic-based and intelligent LLM-powered detection methods with continuous learning from user feedback.

## Features

- **Dual Detection Methods**:
  - **Heuristic**: Fast rule-based detection (<150ms, free)
  - **LLM**: Intelligent contextual analysis (500-2000ms, ~$0.002)
- **25 Cognitive Biases**: Complete implementation of Charlie Munger's framework
- **Learning System**: Improves from user feedback
- **Performance Tracking**: Compare methods with detailed analytics
- **Smart Recommendations**: Suggests optimal method for each text

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd bias-detector-mcp
```

2. Install with uv:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Initialize NLTK data:
```bash
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Claude Desktop Configuration

Add to your Claude Desktop configuration:

### macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
### Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "bias-detector": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.main"],
      "cwd": "/absolute/path/to/bias-detector-mcp",
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Usage

Once configured, the following tools are available in Claude:

### 1. Quick Bias Check (Heuristic)
```
Use detect_bias_heuristic to analyze: "I'm sure our product will succeed because we worked really hard on it"
```

### 2. Deep Analysis (LLM)
```
Use detect_bias_llm to analyze: "Everyone is investing in crypto, so it must be a good investment"
```

### 3. Get Recommendation
```
Use recommend_detection_method for: "Our competitor's product failed, so ours will definitely succeed"
```

### 4. Submit Feedback
```
Use submit_feedback after any analysis to help improve detection accuracy
```

### 5. Check Performance
```
Use get_performance_stats to see how both methods are performing
```

## Example Outputs

### Heuristic Detection
- **Speed**: ~120ms
- **Cost**: $0
- **Best for**: Quick checks, obvious biases, batch processing

### LLM Detection  
- **Speed**: ~1500ms
- **Cost**: ~$0.002
- **Best for**: Nuanced text, complex biases, high-stakes decisions

## Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black src/
```

## License

MIT