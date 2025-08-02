"""
LLM-powered bias detection engine with support for OpenAI, Anthropic, and local models.
"""

import json
import uuid
import time
import os
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum

# LLM provider imports
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

import aiohttp


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMBiasDetector:
    """LLM-powered intelligent bias detection."""
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.cost_tracker = {"daily": 0.0, "monthly": 0.0}
        self.model_costs = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015}
        }
    
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize available LLM providers."""
        providers = {}
        
        # OpenAI
        if openai and os.getenv("OPENAI_API_KEY"):
            providers["openai"] = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Anthropic  
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            providers["anthropic"] = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Local provider always available
        providers["local"] = {"base_url": os.getenv("LOCAL_LLM_URL", "http://localhost:11434")}
        
        return providers
    
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
            # Generate analysis prompt
            prompt = self._build_analysis_prompt(text, context, depth)
            
            # Add retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Call appropriate provider
                    if provider == "openai":
                        result = await self._analyze_with_openai(prompt, model)
                    elif provider == "anthropic":
                        result = await self._analyze_with_anthropic(prompt, model)
                    else:
                        result = await self._analyze_with_local(prompt, model)
                    
                    # Validate response structure
                    if self._validate_llm_response(result):
                        break
                    else:
                        attempts.append(f"Attempt {attempt + 1}: Invalid response structure")
                        if attempt == max_retries - 1:
                            raise ValueError("All attempts produced invalid response structure")
                
                except json.JSONDecodeError as e:
                    attempts.append(f"Attempt {attempt + 1}: JSON decode error: {str(e)}")
                    if attempt == max_retries - 1:
                        raise ValueError(f"JSON parsing failed after {max_retries} attempts: {str(e)}")
                
                except Exception as e:
                    attempts.append(f"Attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
            
            # Parse and validate result
            parsed_result = self._parse_llm_response(result, text, include_reasoning)
            
            # Calculate cost
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
            # Enhanced error response with debugging info
            return {
                "analysis_id": analysis_id,
                "error": f"LLM analysis failed: {str(e)}",
                "method": "llm",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "llm_provider": provider,
                "model_used": model,
                "attempts": attempts,  # For debugging
                "text_length": len(text),
                "context": context
            }
    
    def _build_analysis_prompt(self, text: str, context: str, depth: str) -> str:
        """Build the analysis prompt for the LLM."""
        system_prompt = """You are an expert cognitive bias analyst trained on Charlie Munger's 25 psychological tendencies.

Your task is to analyze text for cognitive biases with high precision and provide actionable insights.

IMPORTANT: You must respond with valid JSON only. No explanatory text outside the JSON structure.

The 25 biases are organized into categories:

DECISION-MAKING BIASES:
4. Doubt-Avoidance Tendency - avoiding uncertainty in decisions
5. Inconsistency-Avoidance Tendency - sticking to previous commitments/beliefs

SOCIAL & EMOTIONAL BIASES:
2. Liking/Loving Tendency - favoring liked entities
3. Disliking/Hating Tendency - unfairly opposing disliked entities  
8. Envy/Jealousy Tendency - resentment over others' advantages
9. Reciprocation Tendency - feeling obligated to return favors
15. Social-Proof Tendency - following what others do

INFORMATION PROCESSING BIASES:
16. Contrast-Misreaction Tendency - misjudging based on comparisons
18. Availability-Misweighing Tendency - overweighting recent/memorable info
22. Authority-Misinfluence Tendency - excessive deference to authority

COGNITIVE BIASES:
12. Excessive Self-Regard Tendency - overestimating own abilities
13. Overoptimism Tendency - unrealistic positive expectations
23. Twaddle Tendency - using meaningless but impressive-sounding words

INCENTIVE & MOTIVATION:
1. Reward and Punishment Super-Response Tendency - overreacting to incentives

SEVERITY GUIDELINES:
1-3: Mild bias with minimal decision impact
4-6: Moderate bias affecting judgment quality  
7-10: Strong bias significantly distorting reasoning

CONFIDENCE GUIDELINES:
0.1-0.4: Weak evidence, possible bias
0.5-0.7: Clear patterns, likely bias
0.8-1.0: Strong evidence, definite bias

Always respond with valid JSON in this exact format:
{
    "biases_detected": [
        {
            "bias_name": "Exact name from Munger's 25 tendencies",
            "bias_id": 1-25,
            "category": "Category from above list",
            "severity": 1-10,
            "confidence": 0.0-1.0,
            "evidence": ["specific quotes from text"],
            "description": "Brief explanation of this bias",
            "reasoning": "Why this bias is present in the text",
            "antidotes": ["specific actionable recommendations"]
        }
    ],
    "lollapalooza_effects": [
        {
            "description": "How multiple biases amplify each other",
            "involved_biases": ["bias1", "bias2"],
            "amplification_factor": 1.0-3.0,
            "risk_level": "low|medium|high|critical"
        }
    ],
    "overall_assessment": {
        "total_biases": number,
        "average_severity": 0.0-10.0,
        "dominant_category": "Primary bias category",
        "risk_score": 0.0-10.0,
        "key_insight": "Most important finding"
    }
}"""

        # Context-specific guidance
        context_guidance = {
            "decision_making": "Focus especially on: Doubt-Avoidance (choosing 'safe' options), Inconsistency-Avoidance (status quo bias), Authority-Misinfluence (deferring to prestige), Availability-Misweighing (recent info bias), and Contrast-Misreaction (comparison bias).",
            "business": "Look for: Reward/Punishment Super-Response (incentive gaming), Excessive Self-Regard (overconfidence), Social-Proof (following competitors), Authority-Misinfluence (consultant bias).",
            "academic": "Examine: Authority-Misinfluence (academic prestige bias), Excessive Self-Regard (overestimating knowledge), Availability-Misweighing (recent studies bias).",
            "personal": "Identify: Liking/Loving (emotional decisions), Social-Proof (peer pressure), Inconsistency-Avoidance (commitment consistency).",
            "general": "Analyze for all bias types with equal attention."
        }

        depth_instructions = {
            "standard": "Focus on the most clear and significant biases with strong evidence.",
            "deep": "Analyze for both obvious and subtle bias patterns, including nuanced interpretations and weak signals.",
            "comprehensive": "Perform exhaustive analysis including edge cases, weak signals, complex interactions, and potential biases that might be developing."
        }

        user_prompt = f"""Analyze the following text for cognitive biases using Charlie Munger's 25 psychological tendencies.

TEXT TO ANALYZE:
"{text}"

CONTEXT: {context}
SPECIFIC GUIDANCE: {context_guidance.get(context, context_guidance['general'])}

ANALYSIS DEPTH: {depth_instructions.get(depth, depth_instructions['standard'])}

Focus on:
1. Identifying specific biases from the 25 tendencies
2. Providing clear evidence from the text with exact quotes
3. Assessing severity (1-10) and confidence (0.0-1.0) accurately
4. Detecting lollapalooza effects (multiple biases amplifying each other)
5. Offering practical, actionable antidotes and mitigation strategies

Return your analysis as a JSON object with the exact structure specified above. No additional text."""

        return system_prompt + "\n\n" + user_prompt
    
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

    async def _analyze_with_openai(self, prompt: str, model: str) -> Dict[str, Any]:
        """Analyze using OpenAI API."""
        if "openai" not in self.providers:
            raise ValueError("OpenAI provider not configured. Set OPENAI_API_KEY environment variable.")
        
        client = self.providers["openai"]
        
        # Split prompt into system and user parts
        parts = prompt.split("\n\n", 1)
        system_content = parts[0] if len(parts) > 1 else ""
        user_content = parts[1] if len(parts) > 1 else prompt
        
        # CRITICAL FIX: Ensure user message explicitly mentions JSON for OpenAI's json_object mode
        if "json" not in user_content.lower():
            user_content += "\n\nPlease return your analysis in the JSON format specified above."
        
        # Get model-specific parameters
        model_params = self._get_model_specific_params(model)
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            temperature=model_params["temperature"],
            max_tokens=model_params.get("max_tokens"),
            top_p=model_params.get("top_p"),
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _analyze_with_anthropic(self, prompt: str, model: str) -> Dict[str, Any]:
        """Analyze using Anthropic API."""
        if "anthropic" not in self.providers:
            raise ValueError("Anthropic provider not configured. Set ANTHROPIC_API_KEY environment variable.")
        
        client = self.providers["anthropic"]
        
        # Get model-specific parameters
        model_params = self._get_model_specific_params(model)
        
        response = await client.messages.create(
            model=model,
            max_tokens=model_params.get("max_tokens", 2048),
            temperature=model_params["temperature"],
            system=prompt.split("\n\n")[0],
            messages=[{"role": "user", "content": prompt.split("\n\n")[1]}]
        )
        
        # Extract JSON from response
        content = response.content[0].text
        # Try to find JSON in the response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(content[json_start:json_end])
        else:
            return json.loads(content)  # Hope it's valid JSON
    
    async def _analyze_with_local(self, prompt: str, model: str) -> Dict[str, Any]:
        """Analyze using local LLM (e.g., Ollama)."""
        base_url = self.providers["local"]["base_url"]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/generate",
                json={
                    "model": model or "llama3.1",
                    "prompt": prompt,
                    "temperature": 0.1,
                    "format": "json"
                }
            ) as response:
                result = await response.json()
                return json.loads(result.get("response", "{}"))
    
    def _validate_llm_response(self, response: Dict[str, Any]) -> bool:
        """Validate LLM response structure."""
        if not isinstance(response, dict):
            return False
            
        required_fields = ["biases_detected", "lollapalooza_effects", "overall_assessment"]
        
        if not all(field in response for field in required_fields):
            return False
        
        # Check biases structure
        biases = response.get("biases_detected", [])
        if not isinstance(biases, list):
            return False
            
        for bias in biases:
            if not isinstance(bias, dict):
                return False
            required_bias_fields = ["bias_name", "severity", "confidence", "evidence"]
            if not all(field in bias for field in required_bias_fields):
                return False
            
            # Validate ranges
            if not (1 <= bias.get("severity", 0) <= 10):
                return False
            if not (0.0 <= bias.get("confidence", -1) <= 1.0):
                return False
        
        # Check overall assessment structure
        assessment = response.get("overall_assessment", {})
        if not isinstance(assessment, dict):
            return False
            
        return True
    
    def _parse_llm_response(self, response: Dict[str, Any], original_text: str, include_reasoning: bool) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        # Ensure required fields exist
        biases_detected = response.get("biases_detected", [])
        lollapalooza_effects = response.get("lollapalooza_effects", [])
        overall_assessment = response.get("overall_assessment", {})
        
        # Validate and clean biases
        validated_biases = []
        for bias in biases_detected:
            validated_bias = {
                "bias_name": bias.get("bias_name", "Unknown Bias"),
                "bias_id": bias.get("bias_id", 0),
                "category": bias.get("category", "Uncategorized"),
                "severity": max(1, min(10, bias.get("severity", 5))),
                "confidence": max(0.0, min(1.0, bias.get("confidence", 0.5))),
                "evidence": bias.get("evidence", []),
                "description": bias.get("description", ""),
                "antidotes": bias.get("antidotes", [])
            }
            
            if include_reasoning and "reasoning" in bias:
                validated_bias["reasoning"] = bias["reasoning"]
            
            validated_biases.append(validated_bias)
        
        # Validate overall assessment
        validated_assessment = {
            "total_biases": len(validated_biases),
            "average_severity": overall_assessment.get("average_severity", 
                sum(b["severity"] for b in validated_biases) / len(validated_biases) if validated_biases else 0),
            "dominant_category": overall_assessment.get("dominant_category", "None"),
            "risk_score": max(0.0, min(10.0, overall_assessment.get("risk_score", 5.0)))
        }
        
        if "key_insight" in overall_assessment:
            validated_assessment["key_insight"] = overall_assessment["key_insight"]
        
        result = {
            "biases_detected": validated_biases,
            "lollapalooza_effects": lollapalooza_effects,
            "overall_assessment": validated_assessment
        }
        
        if include_reasoning:
            result["reasoning"] = response.get("reasoning", "LLM analysis completed")
        
        return result
    
    def _calculate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost for the API call."""
        if provider == "local":
            return 0.0
        
        # Rough token estimation (1 token ≈ 4 characters)
        input_token_count = input_tokens // 4
        output_token_count = output_tokens // 4
        
        if model in self.model_costs:
            costs = self.model_costs[model]
            input_cost = (input_token_count / 1000) * costs["input"]
            output_cost = (output_token_count / 1000) * costs["output"]
            total_cost = input_cost + output_cost
            
            # Track costs
            self.cost_tracker["daily"] += total_cost
            self.cost_tracker["monthly"] += total_cost
            
            return round(total_cost, 6)
        
        return 0.001  # Default estimate