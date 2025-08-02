#!/usr/bin/env python3
"""
Bias Detection MCP Server - Main Entry Point

Dual-interface MCP server providing both heuristic and LLM-powered 
cognitive bias detection using Charlie Munger's 25 psychological tendencies.
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Import our detection engines
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.heuristic_engine import HeuristicBiasDetector
from src.engines.llm_engine import LLMBiasDetector
from src.feedback.learning_system import FeedbackLearningSystem
from src.feedback.performance_tracker import PerformanceTracker
from src.models.analysis_models import (
    BiasAnalysisInput, 
    BiasAnalysisResult, 
    FeedbackData,
    PerformanceStats
)

# Initialize FastMCP server
mcp = FastMCP("Bias Detector Dual Interface")

# Global instances (initialized in main)
heuristic_detector: Optional[HeuristicBiasDetector] = None
llm_detector: Optional[LLMBiasDetector] = None
learning_system: Optional[FeedbackLearningSystem] = None
performance_tracker: Optional[PerformanceTracker] = None


@mcp.tool()
async def detect_bias_heuristic(
    text: str,
    context: str = "general",
    severity_threshold: int = 3,
    include_confidence_scores: bool = True,
    include_antidotes: bool = True
) -> Dict[str, Any]:
    """
    Fast rule-based bias detection using pattern matching and NLP analysis.
    
    Args:
        text: Text to analyze for cognitive biases
        context: Analysis context (business, personal, academic, decision_making, general)
        severity_threshold: Minimum severity level to report (1-10 scale)
        include_confidence_scores: Include confidence scores in results
        include_antidotes: Include bias mitigation recommendations
    
    Returns:
        Comprehensive bias analysis with detected biases and metadata
    """
    start_time = time.time()
    
    try:
        result = await heuristic_detector.analyze({
            "text": text,
            "context": context,
            "severity_threshold": severity_threshold,
            "include_antidotes": include_antidotes
        })
        
        # Add metadata for tracking
        result["method"] = "heuristic"
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        result["estimated_cost"] = 0.0
        result["confidence_scores_included"] = include_confidence_scores
        
        # Track usage
        performance_tracker.record_analysis("heuristic", result)
        
        return result
        
    except Exception as e:
        return {
            "error": f"Heuristic analysis failed: {str(e)}",
            "method": "heuristic",
            "processing_time_ms": (time.time() - start_time) * 1000
        }


@mcp.tool()
async def detect_bias_llm(
    text: str,
    context: str = "general",
    llm_provider: str = "openai",
    model: str = "gpt-4o-mini", 
    analysis_depth: str = "standard",
    include_reasoning: bool = True
) -> Dict[str, Any]:
    """
    LLM-powered intelligent bias detection with deep contextual analysis.
    
    Args:
        text: Text to analyze for cognitive biases
        context: Analysis context
        llm_provider: LLM provider (openai, anthropic, local)
        model: Model name to use
        analysis_depth: Analysis depth (standard, deep, comprehensive)
        include_reasoning: Include detailed reasoning in results
    
    Returns:
        Comprehensive bias analysis with LLM insights and metadata
    """
    start_time = time.time()
    
    try:
        result = await llm_detector.analyze({
            "text": text,
            "context": context,
            "provider": llm_provider,
            "model": model,
            "depth": analysis_depth,
            "include_reasoning": include_reasoning
        })
        
        # Add metadata
        result["method"] = "llm"
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        result["llm_provider"] = llm_provider
        result["model_used"] = model
        
        # Track usage and cost
        performance_tracker.record_analysis("llm", result)
        
        return result
        
    except Exception as e:
        return {
            "error": f"LLM analysis failed: {str(e)}",
            "method": "llm",
            "processing_time_ms": (time.time() - start_time) * 1000
        }


@mcp.tool()
async def submit_feedback(
    analysis_id: str,
    method_used: str,
    accuracy_rating: int,
    usefulness_rating: int,
    missed_biases: List[str] = [],
    false_positives: List[str] = [],
    comments: str = "",
    preferred_method: str = "no_preference"
) -> Dict[str, Any]:
    """
    Submit feedback on bias detection results to improve both engines.
    
    Args:
        analysis_id: ID from previous analysis
        method_used: Method that was used (heuristic or llm)
        accuracy_rating: Accuracy rating (1-10 scale)
        usefulness_rating: Usefulness rating (1-10 scale)
        missed_biases: List of bias names that were missed
        false_positives: List of incorrect detections
        comments: Additional feedback comments
        preferred_method: Preferred method for similar cases
    
    Returns:
        Feedback confirmation with applied improvements
    """
    try:
        feedback_data = {
            "analysis_id": analysis_id,
            "method_used": method_used,
            "accuracy_rating": accuracy_rating,
            "usefulness_rating": usefulness_rating,
            "missed_biases": missed_biases,
            "false_positives": false_positives,
            "comments": comments,
            "preferred_method": preferred_method,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process feedback for learning
        improvements = await learning_system.process_feedback(feedback_data)
        
        # Update performance tracking
        performance_tracker.record_feedback(feedback_data)
        
        return {
            "feedback_id": str(uuid.uuid4()),
            "status": "received",
            "improvements_applied": improvements,
            "thank_you": "Feedback received! This helps improve both detection methods."
        }
        
    except Exception as e:
        return {
            "error": f"Feedback processing failed: {str(e)}",
            "status": "failed"
        }


@mcp.tool()
async def get_performance_stats(
    time_period: str = "last_week",
    include_cost_analysis: bool = True
) -> Dict[str, Any]:
    """
    Get performance comparison and usage statistics for both methods.
    
    Args:
        time_period: Time period for stats (last_24h, last_week, last_month)
        include_cost_analysis: Include detailed cost analysis
    
    Returns:
        Comprehensive performance comparison and recommendations
    """
    try:
        stats = performance_tracker.get_comparative_stats(time_period)
        
        if include_cost_analysis:
            stats["cost_analysis"] = performance_tracker.get_cost_analysis(time_period)
        
        # Add recommendations based on usage patterns
        stats["recommendations"] = learning_system.get_method_recommendations(stats)
        
        return stats
        
    except Exception as e:
        return {
            "error": f"Stats generation failed: {str(e)}",
            "time_period": time_period
        }


@mcp.tool()
async def recommend_detection_method(
    text: str,
    context: str = "general",
    user_priority: str = "balanced"
) -> Dict[str, Any]:
    """
    Get recommendation on which detection method to use based on text analysis.
    
    Args:
        text: Text to analyze
        context: Analysis context
        user_priority: User priority (speed, accuracy, cost, balanced)
    
    Returns:
        Method recommendation with reasoning and expected performance
    """
    try:
        from src.recommendations.usage_optimizer import UsageOptimizer
        
        optimizer = UsageOptimizer(performance_tracker, learning_system)
        recommendation = optimizer.recommend_method(text, context, user_priority)
        
        return recommendation
        
    except Exception as e:
        return {
            "error": f"Recommendation failed: {str(e)}",
            "fallback_recommendation": "heuristic",
            "reasoning": ["Error occurred, defaulting to fast heuristic method"]
        }


def initialize_components():
    """Initialize all detection engines and support systems."""
    global heuristic_detector, llm_detector, learning_system, performance_tracker
    
    print("🔧 Initializing Bias Detection MCP Server...", file=sys.stderr)
    
    try:
        # Initialize detection engines
        heuristic_detector = HeuristicBiasDetector()
        llm_detector = LLMBiasDetector()
        
        # Initialize learning and tracking systems
        learning_system = FeedbackLearningSystem()
        performance_tracker = PerformanceTracker()
        
        print("✅ All components initialized successfully!", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}", file=sys.stderr)
        raise


def main():
    """Main entry point for the MCP server."""
    print("🚀 Starting Bias Detection MCP Server...", file=sys.stderr)
    print("📊 Dual-interface: Heuristic + LLM detection", file=sys.stderr)
    print("🧠 Using Charlie Munger's 25 psychological tendencies", file=sys.stderr)
    
    try:
        # Initialize components synchronously
        initialize_components()
        
        print("🔗 MCP Server ready for connections", file=sys.stderr)
        print("📋 Available tools:", file=sys.stderr)
        print("  - detect_bias_heuristic: Fast rule-based detection", file=sys.stderr)
        print("  - detect_bias_llm: Intelligent LLM-powered detection", file=sys.stderr) 
        print("  - submit_feedback: Improve both engines with feedback", file=sys.stderr)
        print("  - get_performance_stats: Compare method performance", file=sys.stderr)
        print("  - recommend_detection_method: Get smart recommendations", file=sys.stderr)
        
        # Run the MCP server
        mcp.run()
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()