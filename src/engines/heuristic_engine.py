"""
Heuristic-based bias detection engine using pattern matching and NLP analysis.
"""

import re
import uuid
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# NLP imports
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError:
    nltk = None

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class HeuristicBiasDetector:
    """Fast rule-based bias detection using pattern matching and NLP."""
    
    def __init__(self):
        self.bias_database = self._load_bias_database()
        self.pattern_cache = {}
        self._initialize_nltk()
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=1000
        )
        
    def _initialize_nltk(self):
        """Initialize NLTK components if available."""
        if nltk:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.stop_words = set(stopwords.words('english'))
            except:
                self.sentiment_analyzer = None
                self.stop_words = set()
        else:
            self.sentiment_analyzer = None
            self.stop_words = set()
    
    def _load_bias_database(self) -> List[Dict[str, Any]]:
        """Load bias definitions and patterns."""
        # For now, return hardcoded bias definitions based on the bias.txt file
        return [
            {
                "id": 1,
                "name": "Reward and Punishment Super-Response Tendency",
                "category": "Incentive & Motivation",
                "description": "Overreaction to incentive systems, both positive and negative",
                "keywords": ["incentive", "reward", "punishment", "motivation", "bonus", "penalty"],
                "patterns": [
                    r"gaming the system",
                    r"work(?:ing)? to (?:the )?incentives?",
                    r"perverse incentives?",
                    r"motivated by (?:money|rewards?|bonuses?)"
                ],
                "examples": [
                    "Federal Express hourly vs shift pay change",
                    "Xerox salespeople selling old models for better commission"
                ],
                "antidotes": [
                    "Design incentives carefully to align with desired outcomes",
                    "Monitor for gaming behavior regularly",
                    "Consider unintended consequences of reward systems"
                ]
            },
            {
                "id": 2,
                "name": "Liking/Loving Tendency", 
                "category": "Social & Emotional",
                "description": "Bias toward favoring people, products, and actions associated with liked entities",
                "keywords": ["like", "love", "prefer", "favorite", "admire", "respect"],
                "patterns": [
                    r"I (?:really )?like (?:this|that|him|her)",
                    r"because (?:I|we) (?:love|like|prefer)",
                    r"(?:my|our) favorite",
                    r"I admire (?:his|her|their)"
                ],
                "examples": [
                    "Choosing products endorsed by liked celebrities",
                    "Hiring people similar to yourself"
                ],
                "antidotes": [
                    "Separate personal feelings from objective evaluation",
                    "Use structured decision criteria",
                    "Seek diverse perspectives"
                ]
            },
            {
                "id": 3,
                "name": "Disliking/Hating Tendency",
                "category": "Social & Emotional", 
                "description": "Bias against people, products, and actions associated with disliked entities",
                "keywords": ["hate", "dislike", "can't stand", "terrible", "awful", "despise"],
                "patterns": [
                    r"I (?:really )?(?:hate|dislike|can't stand)",
                    r"(?:terrible|awful|horrible) (?:idea|person|company)",
                    r"because (?:I|we) (?:hate|dislike)",
                    r"never trust"
                ],
                "examples": [
                    "Rejecting good ideas from disliked competitors",
                    "Dismissing valid criticism from adversaries"
                ],
                "antidotes": [
                    "Focus on merit rather than source",
                    "Use blind evaluation processes",
                    "Actively seek opposing viewpoints"
                ]
            },
            {
                "id": 12,
                "name": "Excessive Self-Regard Tendency",
                "category": "Self-Perception",
                "description": "Overestimating own abilities, judgment, and contributions",
                "keywords": ["I'm sure", "definitely", "obviously", "clearly", "certainly", "without doubt"],
                "patterns": [
                    r"I'?m (?:absolutely )?(?:sure|certain|confident)",
                    r"(?:definitely|obviously|clearly) (?:will|going to)",
                    r"I (?:always|never) (?:get|make|do)",
                    r"(?:my|our) (?:superior|better|best)",
                    r"no one (?:else )?(?:can|could|would)"
                ],
                "examples": [
                    "90% of Swedish drivers thinking they're above average",
                    "Entrepreneurs overestimating their success probability"
                ],
                "antidotes": [
                    "Seek objective external feedback",
                    "Use base rate information for predictions",
                    "Implement devil's advocate processes"
                ]
            },
            {
                "id": 13,
                "name": "Overoptimism Tendency",
                "category": "Prediction & Planning",
                "description": "Unrealistic positive expectations about future outcomes", 
                "keywords": ["will succeed", "going to work", "bound to", "can't fail", "sure to"],
                "patterns": [
                    r"(?:will|going to) (?:definitely|certainly|surely) (?:succeed|work)",
                    r"(?:bound to|sure to|can't help but) (?:succeed|win|work)",
                    r"(?:easy|simple|no problem) to (?:achieve|reach|get)",
                    r"(?:impossible|can't) (?:fail|go wrong)"
                ],
                "examples": [
                    "Startup founders overestimating success probability",
                    "Project managers underestimating completion time"
                ],
                "antidotes": [
                    "Use reference class forecasting",
                    "Consider failure modes explicitly", 
                    "Seek pessimistic perspectives"
                ]
            },
            {
                "id": 15,
                "name": "Social Proof Tendency", 
                "category": "Social Influence",
                "description": "Following the behavior of others, especially under uncertainty",
                "keywords": ["everyone", "most people", "popular", "trending", "consensus", "majority"],
                "patterns": [
                    r"everyone (?:is|does|thinks|says)",
                    r"most people (?:believe|think|do|choose)",
                    r"(?:very )?popular (?:choice|option|method)",
                    r"(?:general )?consensus (?:is|seems)",
                    r"majority (?:of|believes|thinks)"
                ],
                "examples": [
                    "Choosing crowded restaurants assuming they're better",
                    "Investment bubbles driven by herd behavior"
                ],
                "antidotes": [
                    "Evaluate options independently first",
                    "Question the wisdom of crowds",
                    "Seek contrarian viewpoints"
                ]
            }
        ]
    
    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform heuristic bias analysis."""
        start_time = time.time()
        analysis_id = str(uuid.uuid4())
        
        text = input_data["text"]
        context = input_data.get("context", "general")
        severity_threshold = input_data.get("severity_threshold", 3)
        include_antidotes = input_data.get("include_antidotes", True)
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        segments = self._segment_text(processed_text)
        
        # Detect biases using multiple strategies
        detected_biases = []
        
        for bias_def in self.bias_database:
            matches = self._find_bias_matches(segments, bias_def, context)
            
            if matches:
                bias_instance = self._create_bias_instance(
                    bias_def, matches, include_antidotes
                )
                
                if bias_instance["severity"] >= severity_threshold:
                    detected_biases.append(bias_instance)
        
        # Detect lollapalooza effects
        lollapalooza_effects = self._detect_lollapalooza_effects(detected_biases)
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(detected_biases)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "analysis_id": analysis_id,
            "input_text_length": len(text),
            "biases_detected": detected_biases,
            "lollapalooza_effects": lollapalooza_effects,
            "overall_assessment": overall_assessment,
            "processing_time_ms": processing_time,
            "method": "heuristic",
            "estimated_cost": 0.0
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Basic text cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return text
    
    def _segment_text(self, text: str) -> List[str]:
        """Split text into analyzable segments."""
        if self.sentiment_analyzer and nltk:
            try:
                sentences = sent_tokenize(text)
                return sentences
            except:
                pass
        
        # Fallback to simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_bias_matches(self, segments: List[str], bias_def: Dict, context: str) -> List[Dict[str, Any]]:
        """Find bias pattern matches in text segments."""
        matches = []
        
        # Keyword matching
        keyword_matches = self._find_keyword_matches(segments, bias_def)
        matches.extend(keyword_matches)
        
        # Pattern matching
        pattern_matches = self._find_pattern_matches(segments, bias_def)
        matches.extend(pattern_matches)
        
        # Context-specific matching
        context_matches = self._find_context_matches(segments, bias_def, context)
        matches.extend(context_matches)
        
        return matches
    
    def _find_keyword_matches(self, segments: List[str], bias_def: Dict) -> List[Dict[str, Any]]:
        """Find keyword-based matches."""
        matches = []
        keywords = bias_def.get("keywords", [])
        
        for segment in segments:
            segment_lower = segment.lower()
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in segment_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                matches.append({
                    "type": "keyword",
                    "segment": segment,
                    "matched_keywords": matched_keywords,
                    "confidence": min(0.7, len(matched_keywords) * 0.2)
                })
        
        return matches
    
    def _find_pattern_matches(self, segments: List[str], bias_def: Dict) -> List[Dict[str, Any]]:
        """Find regex pattern matches."""
        matches = []
        patterns = bias_def.get("patterns", [])
        
        for pattern_str in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for segment in segments:
                    if pattern.search(segment):
                        matches.append({
                            "type": "pattern",
                            "segment": segment,
                            "pattern": pattern_str,  
                            "confidence": 0.8
                        })
            except re.error:
                continue  # Skip invalid patterns
        
        return matches
    
    def _find_context_matches(self, segments: List[str], bias_def: Dict, context: str) -> List[Dict[str, Any]]:
        """Find context-specific matches."""
        matches = []
        
        # Context-specific keyword boosting
        if context == "business" and bias_def["id"] in [1, 12, 13]:  # Incentive, self-regard, overoptimism
            business_keywords = ["profit", "revenue", "market", "competition", "strategy", "ROI"]
            for segment in segments:
                segment_lower = segment.lower()
                if any(kw in segment_lower for kw in business_keywords):
                    matches.append({
                        "type": "context",
                        "segment": segment,
                        "context_boost": 0.3,
                        "confidence": 0.6
                    })
        
        return matches
    
    def _create_bias_instance(self, bias_def: Dict, matches: List[Dict], include_antidotes: bool) -> Dict[str, Any]:
        """Create a bias instance from matches."""
        # Calculate severity and confidence
        total_confidence = sum(match["confidence"] for match in matches)
        average_confidence = min(total_confidence / len(matches), 1.0) if matches else 0.0
        
        # Severity based on number of matches and confidence
        severity = min(int(average_confidence * 10) + len(matches), 10)
        
        # Extract evidence
        evidence = [match["segment"] for match in matches[:3]]  # Top 3 pieces of evidence
        
        bias_instance = {
            "bias_name": bias_def["name"],
            "bias_id": bias_def["id"],
            "category": bias_def["category"],
            "severity": severity,
            "confidence": average_confidence,
            "evidence": evidence,
            "description": bias_def["description"],
            "examples": bias_def.get("examples", [])
        }
        
        if include_antidotes:
            bias_instance["antidotes"] = bias_def.get("antidotes", [])
        
        return bias_instance
    
    def _detect_lollapalooza_effects(self, detected_biases: List[Dict]) -> List[Dict[str, Any]]:
        """Detect multiple bias amplification effects."""
        effects = []
        
        if len(detected_biases) >= 2:
            # Simple lollapalooza detection - when multiple biases are present
            high_severity_biases = [b for b in detected_biases if b["severity"] >= 7]
            
            if len(high_severity_biases) >= 2:
                amplification_factor = 1.0 + (len(high_severity_biases) * 0.3)
                
                effects.append({
                    "description": f"Multiple high-severity biases detected, amplifying overall risk",
                    "involved_biases": [b["bias_name"] for b in high_severity_biases],
                    "amplification_factor": amplification_factor,
                    "risk_level": "high" if amplification_factor > 1.5 else "medium"
                })
        
        return effects
    
    def _generate_overall_assessment(self, detected_biases: List[Dict]) -> Dict[str, Any]:
        """Generate overall bias assessment."""
        if not detected_biases:
            return {
                "total_biases": 0,
                "average_severity": 0.0,
                "dominant_category": "None",
                "risk_score": 0.0
            }
        
        total_biases = len(detected_biases)
        average_severity = sum(b["severity"] for b in detected_biases) / total_biases
        
        # Find dominant category
        categories = [b["category"] for b in detected_biases]
        dominant_category = max(set(categories), key=categories.count)
        
        # Calculate risk score (0-10)
        risk_score = min(average_severity * (1 + len(detected_biases) * 0.1), 10.0)
        
        return {
            "total_biases": total_biases,
            "average_severity": round(average_severity, 1),
            "dominant_category": dominant_category,
            "risk_score": round(risk_score, 1)
        }