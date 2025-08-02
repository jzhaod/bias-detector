"""
Usage optimizer for intelligent method recommendations.
"""

from typing import Dict, List, Any, Optional
import re
try:
    from textstat import flesch_reading_ease, lexicon_count, sentence_count
except ImportError:
    # Fallback if textstat not available
    def flesch_reading_ease(text): return 50.0
    def lexicon_count(text): return len(text.split())
    def sentence_count(text): return len(re.findall(r'[.!?]+', text))


class UsageOptimizer:
    """Provides intelligent recommendations on which method to use."""
    
    def __init__(self, performance_tracker, learning_system):
        self.performance_tracker = performance_tracker
        self.learning_system = learning_system
        
    def recommend_method(
        self, 
        text: str, 
        context: str, 
        user_priority: str = "balanced"
    ) -> Dict[str, Any]:
        """Recommend which method to use based on various factors."""
        
        # Analyze text characteristics
        text_analysis = self._analyze_text_complexity(text, context)
        
        # Get historical performance for similar cases
        historical_performance = self._get_similar_case_performance(text_analysis)
        
        # Consider user priorities and constraints
        method_scores = self._calculate_method_scores(
            text_analysis, historical_performance, user_priority
        )
        
        # Generate recommendation
        recommended_method = max(method_scores.items(), key=lambda x: x[1])
        
        return {
            "recommended_method": recommended_method[0],
            "confidence": round(recommended_method[1], 2),
            "reasoning": self._generate_reasoning(method_scores, text_analysis, user_priority),
            "alternative": self._get_alternative_recommendation(method_scores),
            "text_complexity": text_analysis,
            "expected_performance": self._predict_performance(recommended_method[0], text_analysis)
        }
    
    def _analyze_text_complexity(self, text: str, context: str) -> Dict[str, Any]:
        """Analyze text characteristics that affect detection difficulty."""
        
        complexity_indicators = {
            "length": len(text),
            "word_count": lexicon_count(text),
            "reading_ease": flesch_reading_ease(text),
            "sentence_count": sentence_count(text),
            "avg_sentence_length": lexicon_count(text) / max(sentence_count(text), 1),
            "contains_negations": bool(re.search(r'\b(not|no|never|nothing|nobody|nowhere|neither|nor)\b', text.lower())),
            "contains_conditionals": bool(re.search(r'\b(if|unless|provided|assuming|suppose)\b', text.lower())),
            "contains_uncertainty": bool(re.search(r'\b(maybe|perhaps|possibly|might|could|probably)\b', text.lower())),
            "contains_absolutes": bool(re.search(r'\b(always|never|definitely|certainly|absolutely|every|all|none)\b', text.lower())),
            "context": context,
            "estimated_bias_density": self._estimate_bias_density(text)
        }
        
        # Calculate overall complexity score (0-1)
        complexity_score = self._calculate_complexity_score(complexity_indicators)
        complexity_indicators["overall_complexity"] = round(complexity_score, 2)
        
        return complexity_indicators
    
    def _estimate_bias_density(self, text: str) -> float:
        """Estimate density of potential bias indicators."""
        bias_indicators = [
            r'\b(I\'?m sure|definitely|obviously|clearly)\b',
            r'\b(everyone|most people|all|none)\b',
            r'\b(will succeed|can\'t fail|bound to)\b',
            r'\b(love|hate|terrible|amazing)\b',
            r'\b(never|always|impossible|guaranteed)\b'
        ]
        
        indicator_count = 0
        for pattern in bias_indicators:
            indicator_count += len(re.findall(pattern, text.lower()))
        
        # Normalize by text length
        words = len(text.split())
        if words > 0:
            return min(indicator_count / words * 10, 1.0)
        return 0.0
    
    def _calculate_complexity_score(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall complexity score from indicators."""
        score = 0.0
        
        # Length complexity
        if indicators["word_count"] > 200:
            score += 0.2
        elif indicators["word_count"] > 100:
            score += 0.1
        
        # Reading difficulty (lower ease = more complex)
        if indicators["reading_ease"] < 30:
            score += 0.2
        elif indicators["reading_ease"] < 50:
            score += 0.1
        
        # Sentence complexity
        if indicators["avg_sentence_length"] > 25:
            score += 0.15
        
        # Linguistic complexity
        if indicators["contains_negations"]:
            score += 0.1
        if indicators["contains_conditionals"]:
            score += 0.1
        if indicators["contains_uncertainty"]:
            score += 0.15
        
        # Bias density
        score += indicators["estimated_bias_density"] * 0.2
        
        # Context boost
        if indicators["context"] in ["business", "academic", "decision_making"]:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_similar_case_performance(self, text_analysis: Dict) -> Dict[str, float]:
        """Get historical performance for similar text characteristics."""
        # Simplified - in production would query historical data
        # based on text complexity, length, context
        
        complexity = text_analysis["overall_complexity"]
        
        # Mock historical data based on complexity
        if complexity < 0.3:
            return {
                "heuristic_accuracy": 0.85,
                "llm_accuracy": 0.88,
                "heuristic_speed": 120,
                "llm_speed": 1200
            }
        elif complexity < 0.7:
            return {
                "heuristic_accuracy": 0.75,
                "llm_accuracy": 0.92,
                "heuristic_speed": 150,
                "llm_speed": 1500
            }
        else:
            return {
                "heuristic_accuracy": 0.65,
                "llm_accuracy": 0.95,
                "heuristic_speed": 180,
                "llm_speed": 2000
            }
    
    def _calculate_method_scores(
        self, 
        text_analysis: Dict, 
        historical_performance: Dict,
        user_priority: str
    ) -> Dict[str, float]:
        """Calculate scores for each method based on various factors."""
        
        scores = {"heuristic": 0.0, "llm": 0.0}
        
        # Base scores from historical performance
        if historical_performance:
            scores["heuristic"] += historical_performance.get("heuristic_accuracy", 0.7) * 0.3
            scores["llm"] += historical_performance.get("llm_accuracy", 0.8) * 0.3
        
        # Adjust based on text complexity
        complexity = text_analysis["overall_complexity"]
        
        # Heuristic method favored for simpler texts
        scores["heuristic"] += (1.0 - complexity) * 0.4
        
        # LLM method favored for complex texts
        scores["llm"] += complexity * 0.4
        
        # Special case adjustments
        if text_analysis["contains_uncertainty"] or text_analysis["contains_conditionals"]:
            scores["llm"] += 0.1
        
        if text_analysis["word_count"] < 50:
            scores["heuristic"] += 0.1
        
        # Adjust based on user priority
        priority_adjustments = {
            "speed": {"heuristic": 0.3, "llm": -0.2},
            "accuracy": {"heuristic": -0.1, "llm": 0.3},
            "cost": {"heuristic": 0.4, "llm": -0.3},
            "balanced": {"heuristic": 0.0, "llm": 0.0}
        }
        
        adjustments = priority_adjustments.get(user_priority, priority_adjustments["balanced"])
        for method, adjustment in adjustments.items():
            scores[method] += adjustment
        
        # Ensure scores are in valid range
        for method in scores:
            scores[method] = max(0.0, min(1.0, scores[method]))
        
        return scores
    
    def _generate_reasoning(
        self, 
        method_scores: Dict[str, float], 
        text_analysis: Dict, 
        user_priority: str
    ) -> List[str]:
        """Generate human-readable reasoning for the recommendation."""
        
        reasoning = []
        
        recommended = max(method_scores.items(), key=lambda x: x[1])
        complexity = text_analysis["overall_complexity"]
        
        if recommended[0] == "heuristic":
            if complexity < 0.4:
                reasoning.append("Text appears straightforward - heuristic method should handle well")
            if user_priority == "speed":
                reasoning.append("Speed priority favors heuristic method (<150ms vs 500-2000ms)")
            if user_priority == "cost":
                reasoning.append("Cost priority favors heuristic method (free vs $0.001-0.003)")
            if text_analysis["word_count"] < 100:
                reasoning.append("Short text length ideal for pattern matching")
        else:  # llm
            if complexity > 0.6:
                reasoning.append("Complex text with nuanced language - LLM better for subtle patterns")
            if text_analysis["contains_uncertainty"]:
                reasoning.append("Text contains uncertainty markers - LLM better at contextual analysis")
            if text_analysis["contains_conditionals"]:
                reasoning.append("Conditional statements present - LLM can better parse complex logic")
            if user_priority == "accuracy":
                reasoning.append("Accuracy priority favors LLM method for thorough analysis")
        
        # Add confidence reasoning
        score_diff = abs(method_scores["heuristic"] - method_scores["llm"])
        if score_diff < 0.2:
            reasoning.append("Close call - both methods likely to perform similarly")
        elif score_diff > 0.5:
            reasoning.append(f"Strong recommendation - {recommended[0]} significantly better suited")
        
        # Add context-specific reasoning
        if text_analysis["context"] == "business":
            if recommended[0] == "llm":
                reasoning.append("Business context benefits from LLM's nuanced understanding")
        
        return reasoning
    
    def _get_alternative_recommendation(self, method_scores: Dict[str, float]) -> str:
        """Get alternative method recommendation."""
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_methods) > 1:
            alternative = sorted_methods[1][0]
            score_diff = sorted_methods[0][1] - sorted_methods[1][1]
            
            if score_diff < 0.1:
                return f"{alternative} (nearly equal performance)"
            else:
                return alternative
        
        return "none"
    
    def _predict_performance(self, method: str, text_analysis: Dict) -> Dict[str, Any]:
        """Predict expected performance for the recommended method."""
        
        complexity = text_analysis["overall_complexity"]
        word_count = text_analysis["word_count"]
        
        if method == "heuristic":
            # Heuristic predictions
            expected_time = 100 + (word_count * 0.5) + (complexity * 50)
            expected_accuracy = 0.9 - (complexity * 0.3)
            expected_biases = 1 + int(text_analysis["estimated_bias_density"] * 5)
            
            return {
                "expected_response_time_ms": round(expected_time),
                "expected_accuracy": round(expected_accuracy, 2),
                "expected_biases_detected": expected_biases,
                "confidence_level": "high" if complexity < 0.5 else "medium"
            }
        else:  # llm
            # LLM predictions
            expected_time = 800 + (word_count * 2) + (complexity * 400)
            expected_accuracy = 0.85 + (complexity * 0.1)  # Better with complexity
            expected_biases = 2 + int(text_analysis["estimated_bias_density"] * 8)
            
            return {
                "expected_response_time_ms": round(expected_time),
                "expected_accuracy": round(min(expected_accuracy, 0.95), 2),
                "expected_biases_detected": expected_biases,
                "confidence_level": "high" if complexity > 0.3 else "medium",
                "estimated_cost": round(0.001 + (word_count / 1000 * 0.002), 4)
            }