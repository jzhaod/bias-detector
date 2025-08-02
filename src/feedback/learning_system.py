"""
Feedback learning system for continuous improvement of both detection engines.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import re


class FeedbackLearningSystem:
    """Learns from user feedback to improve both detection methods."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/feedback")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_db = defaultdict(list)
        self.pattern_improvements = defaultdict(list)
        self.prompt_optimizations = []
        self.preference_patterns = defaultdict(int)
        
        # Load existing feedback if available
        self._load_feedback_history()
    
    async def process_feedback(self, feedback: Dict[str, Any]) -> List[str]:
        """Process feedback and apply improvements."""
        improvements = []
        
        method = feedback["method_used"]
        
        # Store feedback
        self.feedback_db[method].append(feedback)
        self._save_feedback(feedback)
        
        # Process missed biases
        for missed_bias in feedback.get("missed_biases", []):
            improvement = await self._improve_detection_for_bias(missed_bias, method, feedback)
            if improvement:
                improvements.append(improvement)
        
        # Process false positives
        for false_positive in feedback.get("false_positives", []):
            improvement = await self._reduce_false_positive(false_positive, method, feedback)
            if improvement:
                improvements.append(improvement)
        
        # Learn from preference patterns
        if feedback.get("preferred_method") != "no_preference":
            self._update_preference_patterns(feedback)
            improvements.append(f"Updated preference patterns for {feedback.get('preferred_method')}")
        
        # Update accuracy tracking
        self._update_accuracy_metrics(feedback)
        
        return improvements
    
    async def _improve_detection_for_bias(self, bias_name: str, method: str, feedback: Dict) -> str:
        """Improve detection for a specific bias that was missed."""
        
        if method == "heuristic":
            # Generate new patterns based on context
            new_patterns = self._generate_new_patterns(bias_name, feedback)
            if new_patterns:
                self.pattern_improvements[bias_name].extend(new_patterns)
                self._save_pattern_improvements()
                return f"Added {len(new_patterns)} new detection patterns for {bias_name}"
        
        elif method == "llm":
            # Generate prompt improvements
            prompt_improvement = self._generate_prompt_improvement(bias_name, feedback)
            if prompt_improvement:
                self.prompt_optimizations.append(prompt_improvement)
                self._save_prompt_optimizations()
                return f"Improved LLM prompts for {bias_name} detection"
        
        return ""
    
    async def _reduce_false_positive(self, false_positive: str, method: str, feedback: Dict) -> str:
        """Reduce false positive detections."""
        
        if method == "heuristic":
            # Add exclusion patterns
            exclusion = {
                "bias_name": false_positive,
                "exclusion_context": feedback.get("comments", ""),
                "timestamp": datetime.now().isoformat()
            }
            self.pattern_improvements[f"exclude_{false_positive}"].append(exclusion)
            return f"Added exclusion pattern for {false_positive}"
        
        elif method == "llm":
            # Add negative examples to prompt optimization
            negative_example = {
                "bias_name": false_positive,
                "context": feedback.get("comments", ""),
                "type": "false_positive",
                "timestamp": datetime.now().isoformat()
            }
            self.prompt_optimizations.append(negative_example)
            return f"Added negative example for {false_positive} to prompt optimization"
        
        return ""
    
    def _generate_new_patterns(self, bias_name: str, feedback: Dict) -> List[str]:
        """Generate new detection patterns based on feedback context."""
        patterns = []
        
        # Map bias names to pattern generation strategies
        bias_pattern_generators = {
            "Excessive Self-Regard Tendency": [
                r"I (?:personally )?(?:guarantee|assure)",
                r"(?:trust me|believe me) when I say",
                r"I've (?:never been|always been) (?:wrong|right)"
            ],
            "Overoptimism Tendency": [
                r"(?:absolutely|definitely) no (?:risk|downside)",
                r"(?:guaranteed|certain) (?:success|win)",
                r"(?:can't|won't|impossible to) (?:lose|fail)"
            ],
            "Social Proof Tendency": [
                r"(?:all|most) (?:successful|smart) people",
                r"(?:industry|market) (?:standard|norm)",
                r"(?:widely|universally) (?:accepted|adopted)"
            ]
        }
        
        if bias_name in bias_pattern_generators:
            patterns.extend(bias_pattern_generators[bias_name])
        
        return patterns
    
    def _generate_prompt_improvement(self, bias_name: str, feedback: Dict) -> Dict[str, Any]:
        """Generate prompt improvements for better LLM detection."""
        return {
            "bias_name": bias_name,
            "improvement_type": "missed_detection",
            "context": feedback.get("comments", ""),
            "suggestion": f"Pay special attention to subtle indicators of {bias_name}",
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_preference_patterns(self, feedback: Dict):
        """Update user preference patterns."""
        key = f"{feedback['method_used']}_{feedback['preferred_method']}"
        self.preference_patterns[key] += 1
        
        # Save preference patterns
        self._save_preference_patterns()
    
    def _update_accuracy_metrics(self, feedback: Dict):
        """Update accuracy tracking metrics."""
        metrics_file = self.data_dir / "accuracy_metrics.json"
        
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = defaultdict(lambda: {"total": 0, "sum_accuracy": 0, "sum_usefulness": 0})
            
            method = feedback["method_used"]
            if method not in metrics:
                metrics[method] = {"total": 0, "sum_accuracy": 0, "sum_usefulness": 0}
            
            metrics[method]["total"] += 1
            metrics[method]["sum_accuracy"] += feedback["accuracy_rating"]
            metrics[method]["sum_usefulness"] += feedback["usefulness_rating"]
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error updating accuracy metrics: {e}")
    
    def get_method_recommendations(self, performance_stats: Dict) -> Dict[str, str]:
        """Provide intelligent recommendations based on performance patterns."""
        recommendations = {}
        
        # Analyze performance differences
        heuristic = performance_stats.get("heuristic", {})
        llm = performance_stats.get("llm", {})
        
        h_accuracy = heuristic.get("avg_accuracy_rating", 0)
        l_accuracy = llm.get("avg_accuracy_rating", 0)
        h_speed = heuristic.get("avg_response_time_ms", 0)
        l_speed = llm.get("avg_response_time_ms", 0)
        
        # Cost efficiency recommendation
        if h_accuracy >= l_accuracy * 0.9 and heuristic.get("avg_cost_per_analysis", 0) == 0:
            recommendations["cost_efficiency"] = "Heuristic method provides similar accuracy at zero cost"
        
        # Accuracy recommendation
        if l_accuracy > h_accuracy * 1.2:
            recommendations["accuracy"] = "LLM method shows significantly better accuracy for complex cases"
        
        # Speed recommendation
        if h_speed > 0 and l_speed > 0 and h_speed < l_speed * 0.3:
            recommendations["speed"] = "Heuristic method is much faster for real-time analysis"
        
        # Usage pattern recommendations
        user_preferences = self._analyze_user_preferences()
        if user_preferences:
            recommendations["usage_patterns"] = user_preferences
        
        # Context-specific recommendations
        context_recs = self._get_context_recommendations()
        if context_recs:
            recommendations["context_specific"] = context_recs
        
        return recommendations
    
    def _analyze_user_preferences(self) -> str:
        """Analyze user preference patterns from feedback."""
        total_feedback = sum(len(feedback) for feedback in self.feedback_db.values())
        
        if total_feedback < 5:
            return "Need more feedback to analyze preference patterns"
        
        # Count preferences
        heuristic_prefs = self.preference_patterns.get("heuristic_heuristic", 0) + \
                         self.preference_patterns.get("llm_heuristic", 0)
        llm_prefs = self.preference_patterns.get("llm_llm", 0) + \
                   self.preference_patterns.get("heuristic_llm", 0)
        
        if heuristic_prefs > llm_prefs * 1.5:
            return "User tends to prefer heuristic method - prioritize speed and reliability"
        elif llm_prefs > heuristic_prefs * 1.5:
            return "User tends to prefer LLM method - prioritize accuracy and depth"
        else:
            return "User shows balanced preference - both methods valuable for different scenarios"
    
    def _get_context_recommendations(self) -> str:
        """Get context-specific recommendations."""
        # Analyze feedback by context
        context_performance = defaultdict(lambda: {"heuristic": 0, "llm": 0})
        
        for method, feedbacks in self.feedback_db.items():
            for feedback in feedbacks:
                if "context" in feedback and feedback.get("accuracy_rating", 0) >= 7:
                    context_performance[feedback["context"]][method] += 1
        
        recommendations = []
        for context, counts in context_performance.items():
            if counts["heuristic"] > counts["llm"] * 1.5:
                recommendations.append(f"Heuristic works well for {context} context")
            elif counts["llm"] > counts["heuristic"] * 1.5:
                recommendations.append(f"LLM preferred for {context} context")
        
        return "; ".join(recommendations) if recommendations else ""
    
    def _load_feedback_history(self):
        """Load existing feedback history from disk."""
        feedback_file = self.data_dir / "feedback_history.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_db = defaultdict(list, data.get("feedback_db", {}))
                    self.preference_patterns = defaultdict(int, data.get("preferences", {}))
            except Exception as e:
                print(f"Error loading feedback history: {e}")
    
    def _save_feedback(self, feedback: Dict):
        """Save feedback to disk."""
        feedback_file = self.data_dir / "feedback_history.json"
        try:
            data = {
                "feedback_db": dict(self.feedback_db),
                "preferences": dict(self.preference_patterns)
            }
            with open(feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    def _save_pattern_improvements(self):
        """Save pattern improvements to disk."""
        patterns_file = self.data_dir / "pattern_improvements.json"
        try:
            with open(patterns_file, 'w') as f:
                json.dump(dict(self.pattern_improvements), f, indent=2)
        except Exception as e:
            print(f"Error saving pattern improvements: {e}")
    
    def _save_prompt_optimizations(self):
        """Save prompt optimizations to disk."""
        prompts_file = self.data_dir / "prompt_optimizations.json"
        try:
            with open(prompts_file, 'w') as f:
                json.dump(self.prompt_optimizations, f, indent=2)
        except Exception as e:
            print(f"Error saving prompt optimizations: {e}")
    
    def _save_preference_patterns(self):
        """Save preference patterns to disk."""
        self._save_feedback({})