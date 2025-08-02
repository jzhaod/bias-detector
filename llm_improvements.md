# LLM Bias Detection Improvements

## 1. Fix JSON Format Issue (Priority 1)

### Current Problem:
The "messages must contain the word 'json'" error occurs because OpenAI's JSON mode requires explicit mention of JSON in the conversation.

### Fix in `_analyze_with_openai()`:
```python
async def _analyze_with_openai(self, prompt: str, model: str) -> Dict[str, Any]:
    """Analyze using OpenAI API."""
    if "openai" not in self.providers:
        raise ValueError("OpenAI provider not configured. Set OPENAI_API_KEY environment variable.")
    
    client = self.providers["openai"]
    
    # Fix: Split prompt properly and ensure JSON is mentioned in user message
    parts = prompt.split("\n\n", 1)
    system_content = parts[0] if len(parts) > 1 else ""
    user_content = parts[1] if len(parts) > 1 else prompt
    
    # Ensure user message explicitly mentions JSON requirement
    if "json" not in user_content.lower():
        user_content += "\n\nPlease return your analysis in the JSON format specified above."
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

## 2. Enhanced Prompt Engineering

### Improve System Prompt Structure:
```python
def _build_analysis_prompt(self, text: str, context: str, depth: str) -> str:
    """Build the analysis prompt for the LLM."""
    system_prompt = """You are an expert cognitive bias analyst. Your task is to analyze text for cognitive biases and return results in JSON format.

IMPORTANT: You must respond with valid JSON only. No explanatory text outside the JSON structure.

Based on Charlie Munger's 25 psychological tendencies, identify cognitive biases in the provided text.

BIAS CATEGORIES:
- Decision-Making Biases (doubt-avoidance, inconsistency-avoidance)
- Social & Emotional (liking/loving, envy, social proof)
- Information Processing (availability, contrast, authority)
- Cognitive Biases (overoptimism, excessive self-regard)
- Incentive & Motivation (reward/punishment response)

SEVERITY SCALE:
1-3: Mild bias with minimal impact
4-6: Moderate bias affecting decisions
7-10: Strong bias significantly distorting judgment

CONFIDENCE SCALE:
0.1-0.4: Low confidence (possible bias)
0.5-0.7: Medium confidence (likely bias)
0.8-1.0: High confidence (clear bias evidence)

Return JSON in this exact format:
{
    "biases_detected": [
        {
            "bias_name": "Exact Munger tendency name",
            "bias_id": 1-25,
            "category": "Category from list above",
            "severity": 1-10,
            "confidence": 0.0-1.0,
            "evidence": ["specific quotes from text"],
            "description": "What this bias means",
            "reasoning": "Why this bias is present in the text",
            "antidotes": ["specific actionable recommendations"]
        }
    ],
    "lollapalooza_effects": [
        {
            "description": "How multiple biases interact",
            "involved_biases": ["bias1", "bias2"],
            "amplification_factor": 1.0-3.0,
            "risk_level": "low|medium|high|critical"
        }
    ],
    "overall_assessment": {
        "total_biases": 0,
        "average_severity": 0.0,
        "dominant_category": "Primary category",
        "risk_score": 0.0,
        "key_insight": "Most important finding"
    }
}"""

    # Context-specific instructions
    context_prompts = {
        "decision_making": "Focus on biases affecting decision quality: status quo bias, loss aversion, anchoring, and authority bias.",
        "business": "Look for confirmation bias, overconfidence, social proof, and incentive misalignment.",
        "academic": "Examine authority bias, availability heuristic, and excessive self-regard.",
        "personal": "Identify emotional biases, social proof, and consistency tendencies."
    }
    
    context_instruction = context_prompts.get(context, "Analyze for all types of cognitive biases.")
    
    user_prompt = f"""Analyze this text for cognitive biases:

TEXT: "{text}"

CONTEXT: {context}
INSTRUCTION: {context_instruction}

DEPTH: {depth}

Return your analysis as JSON only - no other text."""

    return system_prompt + "\n\n" + user_prompt
```

## 3. Better Error Handling

### Enhanced Error Recovery:
```python
async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform LLM-powered bias analysis with robust error handling."""
    start_time = time.time()
    analysis_id = str(uuid.uuid4())
    
    text = input_data["text"]
    context = input_data.get("context", "general")
    provider = input_data.get("provider", "openai")
    model = input_data.get("model", "gpt-4o-mini")
    depth = input_data.get("depth", "standard")
    include_reasoning = input_data.get("include_reasoning", True)
    
    # Track attempts for debugging
    attempts = []
    
    try:
        prompt = self._build_analysis_prompt(text, context, depth)
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if provider == "openai":
                    result = await self._analyze_with_openai(prompt, model)
                elif provider == "anthropic":
                    result = await self._analyze_with_anthropic(prompt, model)
                else:
                    result = await self._analyze_with_local(prompt, model)
                
                # Validate result structure
                if self._validate_llm_response(result):
                    break
                else:
                    attempts.append(f"Attempt {attempt + 1}: Invalid response structure")
                    if attempt == max_retries - 1:
                        raise ValueError("All attempts produced invalid responses")
            
            except json.JSONDecodeError as e:
                attempts.append(f"Attempt {attempt + 1}: JSON decode error: {str(e)}")
                if attempt == max_retries - 1:
                    raise ValueError(f"JSON parsing failed after {max_retries} attempts")
            
            except Exception as e:
                attempts.append(f"Attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
        
        # Parse and return successful result
        parsed_result = self._parse_llm_response(result, text, include_reasoning)
        estimated_cost = self._calculate_cost(provider, model, len(prompt), len(str(result)))
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "analysis_id": analysis_id,
            "input_text_length": len(text),
            "biases_detected": parsed_result["biases_detected"],
            "lollapalooza_effects": parsed_result["lollapalooza_effects"],
            "overall_assessment": parsed_result["overall_assessment"],
            "processing_time_ms": processing_time,
            "method": "llm",
            "estimated_cost": estimated_cost,
            "llm_provider": provider,
            "model_used": model,
            "llm_reasoning": parsed_result.get("reasoning") if include_reasoning else None
        }
        
    except Exception as e:
        return {
            "analysis_id": analysis_id,
            "error": f"LLM analysis failed: {str(e)}",
            "method": "llm",
            "processing_time_ms": (time.time() - start_time) * 1000,
            "attempts": attempts,  # For debugging
            "llm_provider": provider,
            "model_used": model
        }

def _validate_llm_response(self, response: Dict[str, Any]) -> bool:
    """Validate LLM response structure."""
    required_fields = ["biases_detected", "lollapalooza_effects", "overall_assessment"]
    
    if not all(field in response for field in required_fields):
        return False
    
    # Check biases structure
    for bias in response.get("biases_detected", []):
        required_bias_fields = ["bias_name", "severity", "confidence", "evidence"]
        if not all(field in bias for field in required_bias_fields):
            return False
    
    return True
```

## 4. Model-Specific Optimizations

### Different approaches for different models:
```python
def _get_model_specific_params(self, model: str) -> Dict[str, Any]:
    """Get model-specific parameters for optimal performance."""
    model_configs = {
        "gpt-4o-mini": {
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.9
        },
        "gpt-4o": {
            "temperature": 0.05,
            "max_tokens": 4096,
            "top_p": 0.8
        },
        "claude-3-haiku-20240307": {
            "temperature": 0.1,
            "max_tokens": 2048
        },
        "claude-3-5-sonnet-20241022": {
            "temperature": 0.05,
            "max_tokens": 4096
        }
    }
    
    return model_configs.get(model, model_configs["gpt-4o-mini"])
```

## 5. Performance Improvements

### Add response caching:
```python
import hashlib
from datetime import datetime, timedelta

class LLMBiasDetector:
    def __init__(self):
        # ... existing initialization ...
        self.response_cache = {}
        self.cache_ttl = timedelta(hours=24)  # Cache responses for 24 hours
    
    def _get_cache_key(self, text: str, context: str, model: str, depth: str) -> str:
        """Generate cache key for response caching."""
        content = f"{text}:{context}:{model}:{depth}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze with caching support."""
        # Check cache first
        cache_key = self._get_cache_key(
            input_data["text"], 
            input_data.get("context", "general"),
            input_data.get("model", "gpt-4o-mini"),
            input_data.get("depth", "standard")
        )
        
        cached_result = self.response_cache.get(cache_key)
        if cached_result and datetime.now() - cached_result["timestamp"] < self.cache_ttl:
            cached_result["analysis_id"] = str(uuid.uuid4())  # New ID
            cached_result["from_cache"] = True
            return cached_result
        
        # ... rest of analysis logic ...
        
        # Cache successful results
        if "error" not in result:
            self.response_cache[cache_key] = {
                **result,
                "timestamp": datetime.now()
            }
        
        return result
```
