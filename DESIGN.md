# Bias Detection MCP Server - Design Document

## Overview

A dual-interface MCP server for cognitive bias detection using Charlie Munger's 25 psychological tendencies framework. Provides both fast heuristic-based and intelligent LLM-powered detection methods with continuous learning from user feedback.

## Architecture

### Core Philosophy
- **User Autonomy**: MCP host chooses the right tool for each situation
- **Transparency**: Clear performance metrics for both approaches  
- **Learning System**: Feedback improves both engines over time
- **Flexibility**: Easy switching and comparison between methods

### Dual-Interface Design

#### Interface 1: Heuristic Detection (`detect_bias_heuristic`)
- **Method**: Rule-based pattern matching + NLP analysis
- **Performance**: <150ms response time, zero cost
- **Accuracy**: 80-85% for clear bias patterns
- **Use Cases**: Quick analysis, batch processing, cost-sensitive scenarios

#### Interface 2: LLM Detection (`detect_bias_llm`)  
- **Method**: LLM-powered contextual analysis
- **Performance**: 500-2000ms response time, $0.001-0.003 per analysis
- **Accuracy**: 95%+ for complex cases
- **Use Cases**: Nuanced text, high-stakes decisions, complex multi-bias scenarios

## API Design

### Tool 1: `detect_bias_heuristic`
Fast rule-based bias detection using pattern matching and NLP analysis.

**Input Schema**:
```json
{
  "text": "string - Text to analyze",
  "context": "enum - business|personal|academic|decision_making|general", 
  "severity_threshold": "number - 1-10 scale, default: 3",
  "include_confidence_scores": "boolean - default: true",
  "include_antidotes": "boolean - default: true"
}
```

### Tool 2: `detect_bias_llm`
LLM-powered intelligent bias detection with deep contextual analysis.

**Input Schema**:
```json
{
  "text": "string - Text to analyze",
  "context": "string - Analysis context",
  "llm_provider": "enum - openai|anthropic|local, default: openai",
  "model": "string - Model name, default: gpt-4o-mini", 
  "analysis_depth": "enum - standard|deep|comprehensive, default: standard",
  "include_reasoning": "boolean - default: true"
}
```

### Tool 3: `submit_feedback`
Provide feedback on bias detection results to improve both engines.

**Input Schema**:
```json
{
  "analysis_id": "string - ID from previous analysis",
  "method_used": "enum - heuristic|llm",
  "accuracy_rating": "number - 1-10 scale",
  "usefulness_rating": "number - 1-10 scale", 
  "missed_biases": "array - List of missed bias names",
  "false_positives": "array - List of incorrect detections",
  "comments": "string - Additional feedback",
  "preferred_method": "enum - heuristic|llm|no_preference"
}
```

### Tool 4: `get_performance_stats`
Get performance comparison and usage statistics for both methods.

**Input Schema**:
```json
{
  "time_period": "enum - last_24h|last_week|last_month, default: last_week",
  "include_cost_analysis": "boolean - default: true"
}
```

### Tool 5: `recommend_detection_method`
Get recommendation on which detection method to use.

**Input Schema**:
```json
{
  "text": "string - Text to analyze",
  "context": "string - Analysis context, default: general",
  "user_priority": "enum - speed|accuracy|cost|balanced, default: balanced"
}
```

## Response Format

### Standard Analysis Result
```json
{
  "analysis_id": "uuid",
  "method": "heuristic|llm",
  "input_text_length": 1247,
  "processing_time_ms": 145.2,
  "estimated_cost": 0.0,
  "biases_detected": [
    {
      "bias_name": "Excessive Self-Regard Tendency",
      "bias_id": 12,
      "category": "Self-Perception Biases",
      "severity": 7,
      "confidence": 0.85,
      "evidence": ["specific text excerpts"],
      "description": "Overestimating own abilities or judgment",
      "antidotes": ["Force objective self-evaluation", "Seek external validation"],
      "examples": ["Swedish drivers study - 90% think above average"]
    }
  ],
  "lollapalooza_effects": [
    {
      "description": "Multiple biases amplifying each other", 
      "involved_biases": ["Authority-Misinfluence", "Social Proof"],
      "amplification_factor": 2.3,
      "risk_level": "high"
    }
  ],
  "overall_assessment": {
    "total_biases": 3,
    "average_severity": 5.2,
    "dominant_category": "Self-Perception Biases",
    "risk_score": 6.8
  }
}
```

## Charlie Munger's 25 Cognitive Biases

### Incentive & Motivation Biases
1. **Reward/Punishment Super-Response** - Overreaction to incentives
2. **Liking/Loving Tendency** - Bias toward favoring liked entities  
3. **Disliking/Hating Tendency** - Bias against disliked entities

### Cognitive Processing Biases
4. **Doubt-Avoidance** - Rush to eliminate uncertainty
5. **Inconsistency-Avoidance** - Resistance to changing beliefs
6. **Curiosity Tendency** - Information seeking behavior
7. **Availability-Misweighting** - Overweighting easily recalled info
8. **Use-It-Or-Lose-It** - Skill atrophy effects

### Social & Comparison Biases  
9. **Kantian Fairness** - Universal rule application
10. **Envy/Jealousy** - Comparative disadvantage focus
11. **Reciprocation** - Favor/disfavor return tendency
12. **Social Proof** - Following crowd behavior
13. **Authority-Misinfluence** - Excessive deference to authority

### Self-Perception Biases
14. **Excessive Self-Regard** - Overestimating own abilities
15. **Overoptimism** - Unrealistic positive expectations  
16. **Simple Pain-Avoiding Denial** - Reality rejection

### Perception & Context Biases
17. **Influence-From-Mere-Association** - Guilt/credit by association
18. **Contrast-Misreaction** - Relative vs absolute judgment errors
19. **Deprival-Super-Reaction** - Loss aversion intensity
20. **Stress-Influence** - Performance changes under pressure

### Aging & Chemical Influences
21. **Drug-Misinfluence** - Chemical cognitive impairment
22. **Senescence-Misinfluence** - Age-related cognitive decline

### Communication Biases
23. **Twaddle Tendency** - Meaningless talk interference
24. **Reason-Respecting** - Accepting poor reasoning
25. **Lollapalooza** - Multiple bias amplification effects

## Learning & Feedback System

### Feedback Loop
1. User analyzes text with chosen method
2. Gets results with performance metadata
3. Provides feedback on accuracy/usefulness
4. System learns and improves both engines  
5. Recommendations get smarter over time
6. User sees performance stats and optimizes usage

### Performance Tracking
- Response time comparison
- Cost analysis and budgeting
- Accuracy ratings from user feedback
- Usage pattern analysis
- Method preference learning

### Intelligent Recommendations
- Text complexity analysis
- Historical performance for similar cases
- User priority consideration (speed/accuracy/cost)
- Contextual method selection
- Continuous optimization

## Implementation Architecture

### Project Structure
```
bias-detector-mcp/
├── src/
│   ├── main.py                    # FastMCP server entry point
│   ├── models/                    # Pydantic data models
│   ├── engines/                   # Detection engines
│   │   ├── heuristic_engine.py   # Rule-based detection
│   │   └── llm_engine.py         # LLM-powered detection
│   ├── feedback/                  # Learning system
│   │   ├── learning_system.py    # Feedback processing
│   │   └── performance_tracker.py# Performance analytics
│   ├── llm/                       # LLM integration
│   │   ├── providers.py          # OpenAI, Anthropic, local
│   │   └── prompts.py            # Specialized prompts
│   └── data/                      # Bias knowledge base
├── data/
│   ├── biases.json               # 25 bias definitions
│   └── patterns.json             # Detection patterns
├── tests/                        # Test suite
└── requirements.txt              # Python dependencies
```

### Technology Stack
- **Framework**: FastMCP (Python MCP server framework)
- **NLP**: NLTK, scikit-learn for text analysis
- **LLM Integration**: OpenAI, Anthropic APIs + local model support
- **Data Models**: Pydantic for type safety
- **Storage**: JSON files for bias knowledge base and feedback data

## Usage Patterns

### Quick Triage
```
recommend_detection_method → detect_bias_heuristic → feedback if needed
```

### High-Stakes Analysis  
```
detect_bias_heuristic → detect_bias_llm → compare results → feedback
```

### Batch Processing
```
detect_bias_heuristic for all → detect_bias_llm for low-confidence cases
```

### Cost-Conscious
```
Start with heuristic, escalate to LLM only when necessary
```

### Learning Mode
```
Try both methods → submit comparative feedback → check stats regularly
```

## Performance Characteristics

### Heuristic Method
- **Speed**: <150ms response time
- **Cost**: $0 (no external API calls)
- **Accuracy**: 80-85% for clear patterns
- **Offline**: Works without internet connection

### LLM Method  
- **Speed**: 500-2000ms response time
- **Cost**: $0.001-0.003 per analysis (model dependent)
- **Accuracy**: 95%+ for complex cases
- **Online**: Requires API access or local model

### Cost Management
- Daily/monthly spending limits
- Usage tracking and alerts
- Model cost optimization
- Smart triggering to minimize costs

## Security & Privacy

### Data Handling
- No persistent storage of analyzed text
- Feedback data stored locally
- Optional local LLM support for sensitive content
- API keys managed through environment variables

### Rate Limiting
- Built-in cost controls
- Request throttling for external APIs
- Graceful fallback when limits exceeded

## Deployment Options

### Claude Desktop Integration
- Standard MCP protocol compliance
- stdio transport for desktop integration
- Simple configuration file setup

### Standalone Server
- HTTP transport option
- REST API endpoints
- Docker container support

## Future Enhancements

### Planned Features
- Custom bias definition support
- Team/organization shared learning
- Advanced analytics dashboard
- Integration with decision frameworks
- Multi-language support

### Extensibility
- Plugin architecture for new bias types
- Custom LLM provider integration
- Advanced feedback mechanisms
- Organizational policy integration