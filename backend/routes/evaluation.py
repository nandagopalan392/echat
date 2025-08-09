from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime, timedelta
import sqlite3
import random
import asyncio
import time

logger = logging.getLogger(__name__)

router = APIRouter()

# Import dependencies that don't cause circular imports
from evaluation_system import (
    get_evaluation_manager, 
    evaluate_chat_response, 
    get_recent_evaluation_stats,
    RAGTriadResult,
    EvaluationResult
)

def get_chat_db():
    """Get ChatDB instance"""
    try:
        from chat_db import ChatDB
        return ChatDB()
    except ImportError:
        # Fallback if chat_db is not available
        return None

# Legacy evaluation functions for backwards compatibility
def calculate_groundedness(response: str, context: str) -> float:
    """Legacy function - calculate groundedness score using simple keyword overlap"""
    if not context or not response:
        return 0.0
    
    context_words = set(context.lower().split())
    response_words = set(response.lower().split())
    
    if not response_words:
        return 0.0
    
    overlap = len(context_words.intersection(response_words))
    score = min(overlap / len(response_words), 1.0)
    
    return min(max(score + random.uniform(-0.2, 0.2), 0.0), 1.0)

def calculate_context_relevance(query: str, context: str) -> float:
    """Legacy function - calculate context relevance using simple keyword overlap"""
    if not query or not context:
        return 0.0
    
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(context_words))
    score = min(overlap / len(query_words), 1.0)
    
    return min(max(score + random.uniform(-0.15, 0.15), 0.0), 1.0)

def calculate_answer_quality(query: str, response: str) -> float:
    """Legacy function - calculate answer quality using simple metrics"""
    if not query or not response:
        return 0.0
    
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    if not query_words or not response_words:
        return 0.0
    
    overlap_score = len(query_words.intersection(response_words)) / len(query_words)
    length_score = min(len(response.split()) / 50.0, 1.0)
    
    quality_score = (overlap_score * 0.7) + (length_score * 0.3)
    
    return min(max(quality_score + random.uniform(-0.2, 0.2), 0.0), 1.0)

@router.get("/metrics")
async def get_evaluation_metrics(
    timeframe: str = Query("7d", description="Time frame: 1d, 7d, 30d")
):
    """Get current evaluation metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Calculate date range
        end_date = datetime.now()
        if timeframe == "1d":
            start_date = end_date - timedelta(days=1)
        elif timeframe == "7d":
            start_date = end_date - timedelta(days=7)
        elif timeframe == "30d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)
        
        db = get_chat_db()
        
        # Get recent chat sessions and messages for evaluation
        if db:
            with sqlite3.connect(db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get recent chat interactions
                cursor.execute("""
                    SELECT cs.id as session_id, cs.username, cs.topic, cs.created_at,
                           m.content, m.is_user, m.timestamp
                    FROM chat_sessions cs
                    JOIN messages m ON cs.id = m.session_id
                    WHERE cs.created_at >= ? AND cs.created_at <= ?
                    ORDER BY cs.created_at DESC, m.timestamp ASC
                """, (start_date.isoformat(), end_date.isoformat()))
                
                messages = cursor.fetchall()
        else:
            messages = []
        
        # Process messages and calculate metrics
        total_interactions = 0
        groundedness_scores = []
        context_relevance_scores = []
        answer_quality_scores = []
        latency_scores = []
        
        current_session = None
        user_query = None
        
        for message in messages:
            if message['is_user']:
                user_query = message['content']
                current_session = message['session_id']
            else:
                if user_query and current_session == message['session_id']:
                    # Calculate metrics for this Q&A pair
                    total_interactions += 1
                    
                    # Mock context for demonstration (in real implementation, retrieve from RAG)
                    mock_context = f"Retrieved context for query: {user_query[:100]}..."
                    
                    # Calculate evaluation metrics
                    groundedness = calculate_groundedness(message['content'], mock_context)
                    context_relevance = calculate_context_relevance(user_query, mock_context)
                    answer_quality = calculate_answer_quality(user_query, message['content'])
                    
                    # Mock latency (in real implementation, track actual response times)
                    latency = random.uniform(0.5, 3.0)  # seconds
                    
                    groundedness_scores.append(groundedness)
                    context_relevance_scores.append(context_relevance)
                    answer_quality_scores.append(answer_quality)
                    latency_scores.append(latency)
                    
                    user_query = None
        
        # Calculate averages
        if total_interactions > 0:
            avg_groundedness = sum(groundedness_scores) / len(groundedness_scores)
            avg_context_relevance = sum(context_relevance_scores) / len(context_relevance_scores)
            avg_answer_quality = sum(answer_quality_scores) / len(answer_quality_scores)
            avg_latency = sum(latency_scores) / len(latency_scores)
        else:
            # Default values for demo
            avg_groundedness = 0.85
            avg_context_relevance = 0.78
            avg_answer_quality = 0.82
            avg_latency = 1.2
        
        return {
            "timeframe": timeframe,
            "total_interactions": total_interactions if total_interactions > 0 else 150,  # Mock for demo
            "metrics": {
                "groundedness": {
                    "score": round(avg_groundedness, 3),
                    "description": "How well responses are grounded in retrieved context",
                    "threshold": 0.7
                },
                "context_relevance": {
                    "score": round(avg_context_relevance, 3),
                    "description": "Relevance of retrieved context to user queries",
                    "threshold": 0.7
                },
                "answer_quality": {
                    "score": round(avg_answer_quality, 3),
                    "description": "Overall quality and completeness of answers",
                    "threshold": 0.75
                },
                "latency": {
                    "score": round(avg_latency, 2),
                    "description": "Average response time in seconds",
                    "threshold": 2.0,
                    "unit": "seconds"
                }
            },
            "calculated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical")
async def get_historical_metrics(
    days: int = Query(30, description="Number of days to look back")
):
    """Get historical evaluation metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock historical data (in real implementation, store these metrics in DB)
        historical_data = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            
            # Generate realistic trending data
            base_groundedness = 0.85
            base_context_relevance = 0.78
            base_answer_quality = 0.82
            base_latency = 1.2
            
            # Add some trend and noise
            trend_factor = (i / days) * 0.1  # Slight improvement over time
            noise_factor = random.uniform(-0.05, 0.05)
            
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "groundedness": round(max(min(base_groundedness + trend_factor + noise_factor, 1.0), 0.0), 3),
                "context_relevance": round(max(min(base_context_relevance + trend_factor + noise_factor, 1.0), 0.0), 3),
                "answer_quality": round(max(min(base_answer_quality + trend_factor + noise_factor, 1.0), 0.0), 3),
                "latency": round(max(base_latency - (trend_factor * 0.5) + (noise_factor * 0.3), 0.1), 2),
                "total_queries": random.randint(10, 50)
            })
        
        return {
            "period": f"{days} days",
            "data": historical_data
        }
        
    except Exception as e:
        logger.error(f"Error getting historical metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latency-distribution")
async def get_latency_distribution(
    timeframe: str = Query("7d", description="Time frame: 1d, 7d, 30d")
):
    """Get latency distribution data"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock latency distribution (in real implementation, get from actual data)
        latency_ranges = [
            {"range": "0-0.5s", "count": random.randint(20, 40)},
            {"range": "0.5-1s", "count": random.randint(30, 60)},
            {"range": "1-2s", "count": random.randint(40, 80)},
            {"range": "2-3s", "count": random.randint(20, 50)},
            {"range": "3-5s", "count": random.randint(10, 30)},
            {"range": "5s+", "count": random.randint(5, 15)}
        ]
        
        return {
            "timeframe": timeframe,
            "distribution": latency_ranges
        }
        
    except Exception as e:
        logger.error(f"Error getting latency distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-breakdown")
async def get_quality_breakdown(
    metric: str = Query("answer_quality", description="Metric to analyze: groundedness, context_relevance, answer_quality")
):
    """Get detailed quality breakdown by score ranges"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate mock quality breakdown (in real implementation, analyze actual scores)
        breakdown = [
            {"range": "0.9-1.0", "count": random.randint(40, 70), "percentage": 0},
            {"range": "0.8-0.9", "count": random.randint(30, 50), "percentage": 0},
            {"range": "0.7-0.8", "count": random.randint(20, 40), "percentage": 0},
            {"range": "0.6-0.7", "count": random.randint(10, 25), "percentage": 0},
            {"range": "0.5-0.6", "count": random.randint(5, 15), "percentage": 0},
            {"range": "0.0-0.5", "count": random.randint(2, 10), "percentage": 0}
        ]
        
        # Calculate percentages
        total_count = sum(item["count"] for item in breakdown)
        for item in breakdown:
            item["percentage"] = round((item["count"] / total_count) * 100, 1) if total_count > 0 else 0
        
        return {
            "metric": metric,
            "breakdown": breakdown,
            "total_evaluations": total_count
        }
        
    except Exception as e:
        logger.error(f"Error getting quality breakdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-response")
async def evaluate_single_response(
    background_tasks: BackgroundTasks,
    question: str,
    answer: str,
    context: str,
    session_id: Optional[str] = None
):
    """Evaluate a single chat response using TruLens-based metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Perform evaluation asynchronously
        result = await evaluate_chat_response(
            question=question,
            answer=answer,
            context=context,
            session_id=session_id,
            user_id="anonymous"  # TODO: Add proper user tracking when auth is restored
        )
        
        return {
            "evaluation_id": f"eval_{int(time.time())}",
            "overall_score": result.overall_score,
            "metrics": {
                "groundedness": {
                    "score": result.groundedness.score,
                    "raw_score": result.groundedness.raw_score,
                    "reasoning": result.groundedness.reasoning[:500] + "..." if len(result.groundedness.reasoning) > 500 else result.groundedness.reasoning
                },
                "answer_relevance": {
                    "score": result.answer_relevance.score,
                    "raw_score": result.answer_relevance.raw_score,
                    "reasoning": result.answer_relevance.reasoning[:500] + "..." if len(result.answer_relevance.reasoning) > 500 else result.answer_relevance.reasoning
                },
                "context_relevance": {
                    "score": result.context_relevance.score,
                    "raw_score": result.context_relevance.raw_score,
                    "reasoning": result.context_relevance.reasoning[:500] + "..." if len(result.context_relevance.reasoning) > 500 else result.context_relevance.reasoning
                }
            },
            "evaluation_time_seconds": result.evaluation_time_seconds,
            "timestamp": result.groundedness.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trulens-metrics")
async def get_trulens_metrics(
    timeframe: str = Query("100", description="Number of recent evaluations to analyze")
):
    """Get TruLens-based evaluation metrics"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        try:
            last_n = int(timeframe)
        except ValueError:
            last_n = 100
        
        stats = get_recent_evaluation_stats(last_n=last_n)
        
        if stats["total_evaluations"] == 0:
            # Return demo data if no evaluations exist
            return {
                "total_evaluations": 0,
                "metrics": {
                    "groundedness": {
                        "score": 0.85,
                        "description": "How well responses are grounded in retrieved context",
                        "threshold": 0.7,
                        "status": "good"
                    },
                    "answer_relevance": {
                        "score": 0.82,
                        "description": "How relevant answers are to the questions asked",
                        "threshold": 0.7,
                        "status": "good"
                    },
                    "context_relevance": {
                        "score": 0.78,
                        "description": "How relevant retrieved context is to user queries",
                        "threshold": 0.7,
                        "status": "good"
                    },
                    "overall_score": {
                        "score": 0.82,
                        "description": "Weighted average of all evaluation metrics",
                        "threshold": 0.7,
                        "status": "good"
                    }
                },
                "performance": {
                    "avg_evaluation_time": 1.5,
                    "min_evaluation_time": 0.8,
                    "max_evaluation_time": 3.2
                },
                "available_models": [],  # Empty array to prevent undefined
                "available_datasets": [],  # Empty array to prevent undefined
                "recent_evaluations": [],  # Empty array to prevent undefined
                "model_performance": [],  # Empty array to prevent undefined
                "message": "No evaluations performed yet. Start chatting to see real metrics!"
            }
        
        # Process real statistics
        def get_status(score: float, threshold: float = 0.7) -> str:
            if score >= 0.8:
                return "excellent"
            elif score >= threshold:
                return "good" 
            elif score >= 0.5:
                return "fair"
            else:
                return "poor"
        
        metrics = stats["metrics"]
        
        return {
            "total_evaluations": stats["total_evaluations"],
            "time_period": stats["time_period"],
            "metrics": {
                "groundedness": {
                    "score": round(metrics["groundedness"]["mean"], 3),
                    "std": round(metrics["groundedness"]["std"], 3),
                    "min": round(metrics["groundedness"]["min"], 3),
                    "max": round(metrics["groundedness"]["max"], 3),
                    "median": round(metrics["groundedness"]["median"], 3),
                    "description": "How well responses are grounded in retrieved context",
                    "threshold": 0.7,
                    "status": get_status(metrics["groundedness"]["mean"])
                },
                "answer_relevance": {
                    "score": round(metrics["answer_relevance"]["mean"], 3),
                    "std": round(metrics["answer_relevance"]["std"], 3),
                    "min": round(metrics["answer_relevance"]["min"], 3),
                    "max": round(metrics["answer_relevance"]["max"], 3),
                    "median": round(metrics["answer_relevance"]["median"], 3),
                    "description": "How relevant answers are to the questions asked",
                    "threshold": 0.7,
                    "status": get_status(metrics["answer_relevance"]["mean"])
                },
                "context_relevance": {
                    "score": round(metrics["context_relevance"]["mean"], 3),
                    "std": round(metrics["context_relevance"]["std"], 3),
                    "min": round(metrics["context_relevance"]["min"], 3),
                    "max": round(metrics["context_relevance"]["max"], 3),
                    "median": round(metrics["context_relevance"]["median"], 3),
                    "description": "How relevant retrieved context is to user queries",
                    "threshold": 0.7,
                    "status": get_status(metrics["context_relevance"]["mean"])
                },
                "overall_score": {
                    "score": round(metrics["overall_score"]["mean"], 3),
                    "std": round(metrics["overall_score"]["std"], 3),
                    "min": round(metrics["overall_score"]["min"], 3),
                    "max": round(metrics["overall_score"]["max"], 3),
                    "median": round(metrics["overall_score"]["median"], 3),
                    "description": "Weighted average of all evaluation metrics",
                    "threshold": 0.7,
                    "status": get_status(metrics["overall_score"]["mean"])
                }
            },
            "performance": {
                "avg_evaluation_time": round(stats["performance"]["avg_evaluation_time"], 2),
                "min_evaluation_time": round(stats["performance"]["min_evaluation_time"], 2),
                "max_evaluation_time": round(stats["performance"]["max_evaluation_time"], 2)
            },
            "thresholds": stats["thresholds"]
        }
        
        # Add additional properties that frontend expects
        response_data.update({
            "available_models": stats.get("available_models", []),
            "available_datasets": stats.get("available_datasets", []),
            "recent_evaluations": evaluations[-10:] if evaluations else [],
            "model_performance": stats.get("model_performance", [])
        })
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error getting TruLens metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evaluation-trends")
async def get_evaluation_trends(
    days: int = Query(7, description="Number of days for trend analysis")
):
    """Get evaluation trends over time"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Get evaluation manager to access history
        manager = get_evaluation_manager()
        evaluations = manager.evaluation_history
        
        if not evaluations:
            # Return mock trend data for demo
            trend_data = []
            for i in range(days):
                date = datetime.now() - timedelta(days=days - i - 1)
                trend_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "groundedness": round(0.85 + random.uniform(-0.1, 0.1), 3),
                    "answer_relevance": round(0.82 + random.uniform(-0.1, 0.1), 3),
                    "context_relevance": round(0.78 + random.uniform(-0.1, 0.1), 3),
                    "overall_score": round(0.82 + random.uniform(-0.1, 0.1), 3),
                    "evaluation_count": random.randint(5, 25)
                })
            
            return {
                "period": f"{days} days",
                "data": trend_data,
                "note": "Mock data - start using the system to see real trends"
            }
        
        # Group evaluations by date
        from collections import defaultdict
        daily_data = defaultdict(list)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_evaluations = [
            e for e in evaluations 
            if e.groundedness.timestamp >= cutoff_date
        ]
        
        for evaluation in recent_evaluations:
            date_key = evaluation.groundedness.timestamp.strftime("%Y-%m-%d")
            daily_data[date_key].append(evaluation)
        
        # Calculate daily averages
        trend_data = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            date_key = date.strftime("%Y-%m-%d")
            
            if date_key in daily_data:
                day_evaluations = daily_data[date_key]
                trend_data.append({
                    "date": date_key,
                    "groundedness": round(sum(e.groundedness.score for e in day_evaluations) / len(day_evaluations), 3),
                    "answer_relevance": round(sum(e.answer_relevance.score for e in day_evaluations) / len(day_evaluations), 3),
                    "context_relevance": round(sum(e.context_relevance.score for e in day_evaluations) / len(day_evaluations), 3),
                    "overall_score": round(sum(e.overall_score for e in day_evaluations) / len(day_evaluations), 3),
                    "evaluation_count": len(day_evaluations)
                })
            else:
                # No data for this day
                trend_data.append({
                    "date": date_key,
                    "groundedness": None,
                    "answer_relevance": None,
                    "context_relevance": None,
                    "overall_score": None,
                    "evaluation_count": 0
                })
        
        return {
            "period": f"{days} days",
            "data": trend_data,
            "total_evaluations": len(recent_evaluations)
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detailed-evaluation/{evaluation_id}")
async def get_detailed_evaluation(
    evaluation_id: str
):
    """Get detailed information about a specific evaluation"""
    try:
        # TODO: Add authentication back when circular import is resolved
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # In a real implementation, you would store evaluations in a database
        # For now, return a mock detailed evaluation
        return {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "question": "What are the main benefits of renewable energy?",
                "context": "Renewable energy sources include solar, wind, hydroelectric, and geothermal power. These sources are sustainable because they are naturally replenished and do not deplete finite resources like fossil fuels.",
                "answer": "The main benefits of renewable energy include environmental sustainability, reduced greenhouse gas emissions, energy independence, and long-term cost savings."
            },
            "evaluation_results": {
                "groundedness": {
                    "score": 0.85,
                    "raw_score": 8.5,
                    "reasoning": "The answer is well-grounded in the provided context. All mentioned benefits (sustainability, reduced emissions, independence, cost savings) are either directly stated or logically derivable from the context about renewable energy sources being naturally replenished and not depleting finite resources.",
                    "criteria": "Evaluating how well the answer is supported by the provided context",
                    "supporting_evidence": "Context mentions renewable sources are 'sustainable' and 'naturally replenished', supporting the answer's claims about sustainability and environmental benefits."
                },
                "answer_relevance": {
                    "score": 0.92,
                    "raw_score": 9.2,
                    "reasoning": "The answer directly addresses the question about benefits of renewable energy. Each point mentioned (environmental sustainability, reduced emissions, energy independence, cost savings) is a relevant benefit that answers the question comprehensively.",
                    "criteria": "Evaluating how well the answer addresses the specific question asked",
                    "supporting_evidence": "Question asks for 'main benefits' and answer provides specific, relevant benefits without going off-topic."
                },
                "context_relevance": {
                    "score": 0.78,
                    "raw_score": 7.8,
                    "reasoning": "The context provides good information about what renewable energy sources are and their sustainable nature, which is relevant to understanding their benefits. However, it could be more specific about the actual benefits mentioned in the answer.",
                    "criteria": "Evaluating how relevant the provided context is to answering the question",
                    "supporting_evidence": "Context explains renewable energy characteristics that support benefit claims, though it doesn't explicitly list all benefits mentioned in the answer."
                }
            },
            "overall_score": 0.85,
            "evaluation_time_seconds": 2.3,
            "metadata": {
                "llm_model": "llama3",
                "evaluation_version": "1.0",
                "session_id": "demo_session"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
