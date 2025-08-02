"""
Pydantic models for bias detection analysis and feedback.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ContextType(str, Enum):
    """Analysis context types."""
    BUSINESS = "business"
    PERSONAL = "personal"
    ACADEMIC = "academic"
    DECISION_MAKING = "decision_making"
    GENERAL = "general"


class BiasAnalysisInput(BaseModel):
    """Input model for bias analysis requests."""
    text: str = Field(description="Text to analyze for cognitive biases")
    context: ContextType = Field(default=ContextType.GENERAL, description="Analysis context")
    severity_threshold: int = Field(default=3, ge=1, le=10, description="Minimum severity to report")
    include_antidotes: bool = Field(default=True, description="Include mitigation recommendations")


class DetectedBias(BaseModel):
    """Model for a detected cognitive bias."""
    bias_name: str = Field(description="Name of the detected bias")
    bias_id: int = Field(description="Munger framework bias ID (1-25)")
    category: str = Field(description="Bias category")
    severity: int = Field(ge=1, le=10, description="Severity rating")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    evidence: List[str] = Field(description="Text evidence supporting detection")
    description: str = Field(description="Bias description")
    antidotes: Optional[List[str]] = Field(default=None, description="Mitigation strategies")
    examples: Optional[List[str]] = Field(default=None, description="Real-world examples")


class LollapaloozaEffect(BaseModel):
    """Model for multiple bias amplification effects."""
    description: str = Field(description="Description of the amplification effect")
    involved_biases: List[str] = Field(description="Names of biases involved")
    amplification_factor: float = Field(description="Amplification multiplier")
    risk_level: str = Field(pattern="^(low|medium|high|critical)$", description="Risk level")


class OverallAssessment(BaseModel):
    """Overall bias analysis assessment."""
    total_biases: int = Field(description="Total number of biases detected")
    average_severity: float = Field(description="Average severity score")
    dominant_category: str = Field(description="Most prevalent bias category")
    risk_score: float = Field(ge=0.0, le=10.0, description="Overall risk score")


class BiasAnalysisResult(BaseModel):
    """Complete bias analysis result."""
    analysis_id: str = Field(description="Unique analysis identifier")
    input_text_length: int = Field(description="Length of analyzed text")
    biases_detected: List[DetectedBias] = Field(description="List of detected biases")
    lollapalooza_effects: List[LollapaloozaEffect] = Field(description="Multi-bias amplification effects")
    overall_assessment: OverallAssessment = Field(description="Overall analysis summary")
    processing_time_ms: float = Field(description="Analysis processing time")
    method: str = Field(description="Detection method used")
    estimated_cost: float = Field(default=0.0, description="Estimated analysis cost")


class FeedbackData(BaseModel):
    """User feedback on analysis results."""
    analysis_id: str = Field(description="Analysis ID being reviewed")
    method_used: str = Field(description="Method that was used")
    accuracy_rating: int = Field(ge=1, le=10, description="Accuracy rating")
    usefulness_rating: int = Field(ge=1, le=10, description="Usefulness rating")
    missed_biases: List[str] = Field(default=[], description="Biases that were missed")
    false_positives: List[str] = Field(default=[], description="Incorrect detections")
    comments: str = Field(default="", description="Additional feedback")
    preferred_method: str = Field(default="no_preference", description="Method preference")
    timestamp: str = Field(description="Feedback timestamp")


class PerformanceMetrics(BaseModel):
    """Performance metrics for a detection method."""
    total_analyses: int = Field(description="Total number of analyses")
    avg_response_time_ms: float = Field(description="Average response time")
    median_response_time_ms: float = Field(description="Median response time")
    total_cost: float = Field(description="Total cost incurred")
    avg_cost_per_analysis: float = Field(description="Average cost per analysis")
    avg_confidence: float = Field(description="Average confidence score")
    avg_biases_detected: float = Field(description="Average biases per analysis")
    feedback_count: int = Field(description="Number of feedback responses")
    avg_accuracy_rating: Optional[float] = Field(description="Average accuracy rating from feedback")
    avg_usefulness_rating: Optional[float] = Field(description="Average usefulness rating from feedback")


class PerformanceComparison(BaseModel):
    """Comparison between detection methods."""
    winner: str = Field(description="Better performing method")
    ratio: str = Field(description="Performance ratio")
    description: str = Field(description="Comparison description")


class PerformanceStats(BaseModel):
    """Complete performance statistics."""
    time_period: str = Field(description="Time period covered")
    heuristic: Optional[PerformanceMetrics] = Field(description="Heuristic method stats")
    llm: Optional[PerformanceMetrics] = Field(description="LLM method stats")
    comparison: Dict[str, PerformanceComparison] = Field(description="Method comparisons")
    cost_analysis: Optional[Dict[str, Any]] = Field(description="Cost analysis details")
    recommendations: Dict[str, str] = Field(description="Usage recommendations")


class TextComplexityAnalysis(BaseModel):
    """Analysis of text complexity for method recommendation."""
    length: int = Field(description="Text length in characters")
    word_count: int = Field(description="Number of words")
    reading_ease: float = Field(description="Flesch reading ease score")
    sentence_complexity: int = Field(description="Number of sentences")
    contains_negations: bool = Field(description="Contains negation words")
    contains_conditionals: bool = Field(description="Contains conditional statements")
    contains_uncertainty: bool = Field(description="Contains uncertainty markers")
    context: str = Field(description="Analysis context")
    estimated_bias_density: float = Field(description="Estimated bias pattern density")
    overall_complexity: float = Field(ge=0.0, le=1.0, description="Overall complexity score")


class MethodRecommendation(BaseModel):
    """Recommendation for which detection method to use."""
    recommended_method: str = Field(description="Recommended detection method")
    confidence: float = Field(ge=0.0, le=1.0, description="Recommendation confidence")
    reasoning: List[str] = Field(description="Reasons for recommendation")
    alternative: str = Field(description="Alternative method suggestion")
    text_complexity: TextComplexityAnalysis = Field(description="Text complexity analysis")
    expected_performance: Dict[str, Any] = Field(description="Expected performance metrics")