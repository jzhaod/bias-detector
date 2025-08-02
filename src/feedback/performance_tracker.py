"""
Performance tracking and comparison system for both detection methods.
"""

import json
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


class PerformanceTracker:
    """Tracks and compares performance of both detection methods."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/performance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.analysis_history = defaultdict(list)
        self.feedback_history = defaultdict(list)
        
        # Load existing data
        self._load_performance_data()
    
    def record_analysis(self, method: str, result: Dict[str, Any]):
        """Record analysis results for performance tracking."""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": result.get("analysis_id"),
            "method": method,
            "processing_time_ms": result.get("processing_time_ms", 0),
            "biases_detected": len(result.get("biases_detected", [])),
            "confidence_scores": [b.get("confidence", 0) for b in result.get("biases_detected", [])],
            "estimated_cost": result.get("estimated_cost", 0),
            "text_length": result.get("input_text_length", 0),
            "error": result.get("error") is not None
        }
        
        self.analysis_history[method].append(record)
        
        # Keep only recent history (last 30 days)
        self._trim_old_records()
        
        # Save to disk
        self._save_performance_data()
    
    def record_feedback(self, feedback_data: Dict[str, Any]):
        """Record user feedback for performance analysis."""
        
        method = feedback_data["method_used"]
        self.feedback_history[method].append(feedback_data)
        
        # Trim old feedback
        self._trim_old_feedback()
        
        # Save to disk
        self._save_performance_data()
    
    def get_comparative_stats(self, time_period: str) -> Dict[str, Any]:
        """Generate comparative performance statistics."""
        
        # Determine time range
        cutoff = self._get_cutoff_time(time_period)
        
        stats = {}
        
        for method in ["heuristic", "llm"]:
            method_records = [
                r for r in self.analysis_history[method]
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]
            
            if not method_records:
                stats[method] = {
                    "status": "no_data",
                    "message": f"No {method} analyses in {time_period}"
                }
                continue
            
            # Calculate performance metrics
            response_times = [r["processing_time_ms"] for r in method_records if not r.get("error")]
            costs = [r["estimated_cost"] for r in method_records]
            confidence_scores = [
                score for r in method_records 
                for score in r["confidence_scores"] 
                if not r.get("error")
            ]
            
            # Get feedback metrics
            feedback_records = [
                f for f in self.feedback_history[method]
                if datetime.fromisoformat(f["timestamp"]) > cutoff
            ]
            
            accuracy_ratings = [f["accuracy_rating"] for f in feedback_records]
            usefulness_ratings = [f["usefulness_rating"] for f in feedback_records]
            
            # Calculate statistics
            stats[method] = {
                "total_analyses": len(method_records),
                "successful_analyses": len([r for r in method_records if not r.get("error")]),
                "error_rate": len([r for r in method_records if r.get("error")]) / len(method_records) if method_records else 0,
                
                # Performance metrics
                "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
                "median_response_time_ms": statistics.median(response_times) if response_times else 0,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                
                # Cost metrics
                "total_cost": sum(costs),
                "avg_cost_per_analysis": statistics.mean(costs) if costs else 0,
                
                # Quality metrics
                "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
                "avg_biases_detected": statistics.mean([r["biases_detected"] for r in method_records]),
                
                # Feedback-based metrics
                "feedback_count": len(feedback_records),
                "avg_accuracy_rating": statistics.mean(accuracy_ratings) if accuracy_ratings else None,
                "avg_usefulness_rating": statistics.mean(usefulness_ratings) if usefulness_ratings else None,
                
                # Usage patterns
                "peak_usage_hour": self._get_peak_usage_hour(method_records),
                "avg_text_length": statistics.mean([r["text_length"] for r in method_records if r["text_length"] > 0])
            }
        
        # Add comparative insights
        stats["comparison"] = self._generate_comparison_insights(stats)
        
        return stats
    
    def get_cost_analysis(self, time_period: str) -> Dict[str, Any]:
        """Get detailed cost analysis."""
        
        cutoff = self._get_cutoff_time(time_period)
        
        # Calculate costs by method
        method_costs = {}
        
        for method in ["heuristic", "llm"]:
            records = [
                r for r in self.analysis_history[method]
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]
            
            costs = [r["estimated_cost"] for r in records]
            
            method_costs[method] = {
                "total_cost": sum(costs),
                "avg_cost": statistics.mean(costs) if costs else 0,
                "analysis_count": len(records),
                "cost_per_1k_chars": (sum(costs) / sum(r["text_length"] for r in records) * 1000) if records and sum(r["text_length"] for r in records) > 0 else 0
            }
        
        # Calculate cost comparison
        if method_costs["heuristic"]["avg_cost"] > 0:
            cost_ratio = method_costs["llm"]["avg_cost"] / method_costs["heuristic"]["avg_cost"]
        else:
            cost_ratio = float('inf') if method_costs["llm"]["avg_cost"] > 0 else 1.0
        
        return {
            "method_costs": method_costs,
            "cost_ratio": cost_ratio,
            "total_spent": sum(m["total_cost"] for m in method_costs.values()),
            "recommendation": self._get_cost_recommendation(method_costs, cost_ratio)
        }
    
    def _generate_comparison_insights(self, stats: Dict) -> Dict[str, Any]:
        """Generate insights comparing both methods."""
        
        insights = {}
        
        heuristic = stats.get("heuristic", {})
        llm = stats.get("llm", {})
        
        # Skip if either has no data
        if heuristic.get("status") == "no_data" or llm.get("status") == "no_data":
            return {"status": "insufficient_data"}
        
        # Speed comparison
        h_speed = heuristic.get("avg_response_time_ms", float('inf'))
        l_speed = llm.get("avg_response_time_ms", float('inf'))
        
        if h_speed < float('inf') and l_speed < float('inf'):
            speed_ratio = l_speed / h_speed if h_speed > 0 else float('inf')
            insights["speed"] = {
                "winner": "heuristic" if h_speed < l_speed else "llm",
                "ratio": f"{speed_ratio:.1f}x",
                "description": f"{'Heuristic' if h_speed < l_speed else 'LLM'} is {speed_ratio:.1f}x faster"
            }
        
        # Cost comparison
        h_cost = heuristic.get("avg_cost_per_analysis", 0)
        l_cost = llm.get("avg_cost_per_analysis", 0)
        
        if l_cost > 0:
            cost_ratio = l_cost / h_cost if h_cost > 0 else float('inf')
            insights["cost"] = {
                "heuristic_cost": h_cost,
                "llm_cost": l_cost,
                "ratio": f"{cost_ratio:.0f}x" if cost_ratio < float('inf') else "∞x",
                "description": f"LLM costs {cost_ratio:.0f}x more per analysis" if cost_ratio < float('inf') else "LLM has cost, heuristic is free"
            }
        
        # Accuracy comparison (from user feedback)
        h_accuracy = heuristic.get("avg_accuracy_rating")
        l_accuracy = llm.get("avg_accuracy_rating")
        
        if h_accuracy and l_accuracy:
            insights["accuracy"] = {
                "heuristic": round(h_accuracy, 1),
                "llm": round(l_accuracy, 1),
                "winner": "heuristic" if h_accuracy > l_accuracy else "llm" if l_accuracy > h_accuracy else "tie",
                "difference": round(abs(h_accuracy - l_accuracy), 1)
            }
        
        # Detection rate comparison
        h_detection = heuristic.get("avg_biases_detected", 0)
        l_detection = llm.get("avg_biases_detected", 0)
        
        insights["detection_rate"] = {
            "heuristic": round(h_detection, 1),
            "llm": round(l_detection, 1),
            "description": f"{'LLM' if l_detection > h_detection else 'Heuristic'} detects {abs(l_detection - h_detection):.1f} more biases on average"
        }
        
        # Reliability comparison
        h_error = heuristic.get("error_rate", 0)
        l_error = llm.get("error_rate", 0)
        
        insights["reliability"] = {
            "heuristic_error_rate": f"{h_error * 100:.1f}%",
            "llm_error_rate": f"{l_error * 100:.1f}%",
            "more_reliable": "heuristic" if h_error < l_error else "llm" if l_error < h_error else "equal"
        }
        
        return insights
    
    def _get_cutoff_time(self, time_period: str) -> datetime:
        """Get cutoff time for the specified period."""
        now = datetime.now()
        
        if time_period == "last_24h":
            return now - timedelta(hours=24)
        elif time_period == "last_week":
            return now - timedelta(weeks=1)
        else:  # last_month
            return now - timedelta(days=30)
    
    def _get_peak_usage_hour(self, records: List[Dict]) -> int:
        """Find the hour with most analyses."""
        if not records:
            return 0
        
        hours = [datetime.fromisoformat(r["timestamp"]).hour for r in records]
        if hours:
            return max(set(hours), key=hours.count)
        return 0
    
    def _get_cost_recommendation(self, method_costs: Dict, cost_ratio: float) -> str:
        """Generate cost-based recommendation."""
        if cost_ratio == float('inf'):
            return "Consider heuristic method for cost-sensitive applications"
        elif cost_ratio > 100:
            return "LLM is significantly more expensive - use sparingly for complex cases"
        elif cost_ratio > 10:
            return "Balance cost vs accuracy needs - use LLM for high-value decisions"
        else:
            return "Cost difference is reasonable - choose based on accuracy needs"
    
    def _trim_old_records(self):
        """Remove records older than 30 days."""
        cutoff = datetime.now() - timedelta(days=30)
        
        for method in self.analysis_history:
            self.analysis_history[method] = [
                r for r in self.analysis_history[method]
                if datetime.fromisoformat(r["timestamp"]) > cutoff
            ]
    
    def _trim_old_feedback(self):
        """Remove feedback older than 30 days."""
        cutoff = datetime.now() - timedelta(days=30)
        
        for method in self.feedback_history:
            self.feedback_history[method] = [
                f for f in self.feedback_history[method]
                if datetime.fromisoformat(f["timestamp"]) > cutoff
            ]
    
    def _load_performance_data(self):
        """Load performance data from disk."""
        perf_file = self.data_dir / "performance_history.json"
        
        if perf_file.exists():
            try:
                with open(perf_file, 'r') as f:
                    data = json.load(f)
                    self.analysis_history = defaultdict(list, data.get("analysis_history", {}))
                    self.feedback_history = defaultdict(list, data.get("feedback_history", {}))
            except Exception as e:
                print(f"Error loading performance data: {e}")
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        perf_file = self.data_dir / "performance_history.json"
        
        try:
            data = {
                "analysis_history": dict(self.analysis_history),
                "feedback_history": dict(self.feedback_history)
            }
            with open(perf_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")