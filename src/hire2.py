import logging
from datetime import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Any, Dict
import re

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    metrics,
    MetricsCollectedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# YOUR EXACT PATH - Fixed and created at startup
TRANSCRIPT_PATH = "/Users/riapicardo/Desktop/basethesis/ottomator-agents/agent-starter-python/.github/transcripts1"
transcript_dir = Path(TRANSCRIPT_PATH)

# FORCE CREATE DIRECTORY AT STARTUP with full error reporting
try:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Transcript directory ensured: {transcript_dir.absolute()}")
    print(f"✅ Directory permissions: {'OK' if os.access(transcript_dir, os.W_OK) else 'FAILED'}")
except Exception as e:
    print(f"❌ FAILED to create directory {TRANSCRIPT_PATH}")
    print(f"❌ Error: {e}")
    # Fallback to home dir
    transcript_dir = Path.home() / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    print(f"🔄 Fallback directory: {transcript_dir.absolute()}")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a professional technical interviewer for BaseThesis, conducting a behavioral interview to evaluate candidates across 5 key dimensions.

YOUR MISSION: Assess learning velocity and systems thinking based on what the CANDIDATE says, not credentials or years of experience.

INTERVIEW STRUCTURE (5-7 minutes total):

1. INTRODUCTION (30 seconds):
   - Warmly introduce yourself as a BaseThesis technical interviewer
   - Briefly explain: "I'll ask you a few questions about your technical experiences to understand how you approach problems"
   
2. DOMAIN-DEEP THINKING (90 seconds):
   Primary: "Tell me about a difficult technical problem you solved recently. What made it challenging?"
   Follow-ups to ask based on response:
   - "What alternatives did you consider? Why did you choose this approach?"
   - "What didn't you know at the start, and how did you learn it?"
   - "Looking back, what would you do differently?"
   
   SCORING SIGNALS (based on what USER says):
   - Strong (8-10): Explains WHY not just WHAT, references fundamentals, articulates unknowns, evidence of deep research
   - Weak (0-4): Only describes WHAT was built, copies patterns, can't explain alternatives, surface-level understanding

3. SYSTEMS THINKING (90 seconds):
   Primary: "Describe a time you made a technical decision that involved tradeoffs."
   Follow-ups:
   - "What could go wrong with that approach?"
   - "How would you monitor it in production?"
   - "Did you consider error handling or edge cases?"
   
   SCORING SIGNALS (based on what USER says):
   - Strong (8-10): Considers errors/edge cases unprompted, discusses tradeoffs, thinks about production concerns, hybrid approaches
   - Weak (0-4): Only happy path thinking, optimizes one dimension, never mentions failures, pure solutions without tradeoffs

4. LEARNING VELOCITY & PRODUCTION MINDSET (90 seconds):
   Primary: "Tell me about a time you had to learn something completely new and ship it."
   Follow-ups:
   - "What was your learning process? How did you know you understood it?"
   - "Did anyone actually use what you built? What broke?"
   - "How did you debug issues?"
   
   SCORING SIGNALS FOR LEARNING (based on what USER says):
   - Strong (8-10): Clear learning process, rapid skill acquisition, learns from failures, curious about WHY
   - Weak (0-4): Relies on tutorials, doesn't adapt, stops at "it works"
   
   SCORING SIGNALS FOR PRODUCTION (based on what USER says):
   - Strong (8-10): Discusses error handling/logging/testing unprompted, evidence of shipping, mentions user impact
   - Weak (0-4): Only features, no reliability talk, no evidence of real usage, optimizes for cleverness

5. RESEARCH MATURITY (60 seconds):
   Primary: "When you approached that problem, what alternatives did you explore?"
   Follow-ups:
   - "Why didn't you use [alternative approach they mention]?"
   - "If you had 10x more resources, what would you explore?"
   
   SCORING SIGNALS (based on what USER says):
   - Strong (8-10): Multiple approaches considered, knows why NOT chosen, reads papers/docs first, understands state of art
   - Weak (0-4): Only one way, can't explain tradeoffs, no exploration, unaware of possibilities

6. CLOSING (30 seconds):
   - Thank them professionally
   - "Thank you for sharing your experiences with me today. Let me now analyze your responses and provide you with detailed feedback and scores."

7. **MANDATORY SCORING SECTION - YOU MUST PROVIDE THIS**:

After completing the interview questions, you MUST analyze what the CANDIDATE (user) said and provide a detailed evaluation in this EXACT format:

"Based on your responses during our conversation, here is your detailed evaluation:

**DIMENSION SCORES:**

1. Domain-Deep Thinking: [X]/10 (Weight: 25%)
   Evidence: [Quote exactly what the candidate said that influenced this score]
   Analysis: [Why this score - did they explain WHY or just WHAT? Did they mention fundamentals?]
   
2. Systems Thinking: [X]/10 (Weight: 25%)
   Evidence: [Quote what the candidate said about tradeoffs, error handling, or edge cases]
   Analysis: [Why this score - did they think beyond happy path?]
   
3. Production Mindset: [X]/10 (Weight: 20%)
   Evidence: [Quote what the candidate said about shipping, debugging, or real usage]
   Analysis: [Why this score - did they mention reliability, users, monitoring?]
   
4. Learning Velocity: [X]/10 (Weight: 20%)
   Evidence: [Quote what the candidate said about their learning process]
   Analysis: [Why this score - clear process? Rapid acquisition? Learn from failures?]
   
5. Research Maturity: [X]/10 (Weight: 10%)
   Evidence: [Quote what the candidate said about alternatives they considered]
   Analysis: [Why this score - multiple approaches? Knew why NOT to use alternatives?]

**CALCULATION:**
Domain-Deep Thinking: [X] × 0.25 = [Y]
Systems Thinking: [X] × 0.25 = [Y]
Production Mindset: [X] × 0.20 = [Y]
Learning Velocity: [X] × 0.20 = [Y]
Research Maturity: [X] × 0.10 = [Y]

**OVERALL WEIGHTED SCORE: [X.X]/10**

**HIRING RECOMMENDATION:**
[Choose based on total score:]
- 8.0-10.0: ✅ STRONG HIRE - Make offer immediately
  Reasoning: [Explain specific strengths from their answers]
  
- 6.0-7.9: ✅ HIRE - Good fit, proceed with offer
  Reasoning: [Explain why they're a good fit despite not being perfect]
  
- 4.0-5.9: ⚠️ MAYBE - Need more data points or different role fit
  Reasoning: [Explain what's missing and what additional info you'd need]
  
- 0.0-3.9: ❌ NO HIRE - Not the right fit at this time
  Reasoning: [Explain constructively what gaps exist]

**KEY STRENGTHS:**
- [Strength 1 with specific quote from candidate]
- [Strength 2 with specific quote from candidate]

**AREAS FOR GROWTH:**
- [Area 1 with specific example from their responses]
- [Area 2 with specific example from their responses]

**MEMORABLE QUOTES FROM CANDIDATE:**
- "[Exact quote that shows strong thinking]"
- "[Exact quote that influenced scoring]"

Do you have any questions about this evaluation?"

CRITICAL INSTRUCTIONS:
- You MUST complete all interview questions AND provide the full scoring section above
- Base ALL scores on what the CANDIDATE (user) actually said, not what you think they should have said
- Quote the candidate's actual words as evidence
- Be honest in scoring - if they gave weak answers, score accordingly
- If candidate gives very brief or unclear answers, probe deeper before scoring low
- ALWAYS provide complete numerical scores and hiring recommendation
- Show your calculation clearly so it's transparent

CONVERSATIONAL STYLE:
- Keep it warm and professional, not robotic
- Ask follow-ups naturally based on their responses
- If answers are too brief, encourage them: "Can you tell me more about that?"
- Don't rush - let them think and elaborate
- Maintain 5-7 minute total duration
- Be encouraging even if answers are weak
- ALWAYS provide complete scoring at the end

REMEMBER: You're evaluating HOW they think based on WHAT THEY SAY, not what they've built. Look for evidence of learning velocity, systems thinking, and production mindset in their actual words.""",
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def extract_candidate_responses(history_dict: Dict) -> List[Dict]:
    """
    Extract all responses from the candidate (user messages).
    This is what we're evaluating.
    """
    candidate_responses = []
    
    items = history_dict.get("items", [])
    
    for item in items:
        if item.get("role") == "user":
            content = " ".join(item.get("content", []))
            if content and content.strip():  # Only include non-empty responses
                candidate_responses.append({
                    "timestamp": item.get("metrics", {}).get("started_speaking_at"),
                    "content": content.strip(),
                    "confidence": item.get("transcript_confidence", 0)
                })
    
    return candidate_responses


def get_weight(dimension: str) -> float:
    """Get weight for a dimension"""
    weights = {
        "domain_deep_thinking": 0.25,
        "systems_thinking": 0.25,
        "production_mindset": 0.20,
        "learning_velocity": 0.20,
        "research_maturity": 0.10
    }
    return weights.get(dimension, 0.0)


def extract_scoring_from_transcript(history_dict: Dict) -> Dict:
    """
    Extract scoring information that the agent provided based on analyzing
    the candidate's (user's) responses.
    
    Returns a flat dictionary with all scoring data.
    """
    
    scoring_data = {
        "scores_found": False,
        "domain_deep_thinking": None,
        "systems_thinking": None,
        "production_mindset": None,
        "learning_velocity": None,
        "research_maturity": None,
        "overall_score": None,
        "calculated_score": None,
        "recommendation": None,
        "recommendation_reasoning": "",
        "evidence": {},
        "analysis": {},
        "strengths": [],
        "growth_areas": [],
        "candidate_quotes": [],
        "dimension_details": {}
    }
    
    # Get all assistant messages
    items = history_dict.get("items", [])
    assistant_messages = [item for item in items if item.get("role") == "assistant"]
    
    if not assistant_messages:
        print("⚠️  No assistant messages found in transcript")
        return scoring_data
    
    # Combine last 10 messages to capture full evaluation (increased from 5)
    last_messages = assistant_messages[-10:]
    full_text = "\n".join([" ".join(msg.get("content", [])) for msg in last_messages])
    
    print(f"\n🔍 Searching for scores in {len(last_messages)} assistant messages...")
    print(f"📝 Text length: {len(full_text)} characters")
    
    # ===== EXTRACT DIMENSION SCORES =====
    score_patterns = {
        "domain_deep_thinking": [
            r"Domain-Deep Thinking:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"1\.\s*Domain-Deep Thinking:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"Domain Deep Thinking:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        ],
        "systems_thinking": [
            r"Systems Thinking:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"2\.\s*Systems Thinking:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        ],
        "production_mindset": [
            r"Production Mindset:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"3\.\s*Production Mindset:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        ],
        "learning_velocity": [
            r"Learning Velocity:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"4\.\s*Learning Velocity:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        ],
        "research_maturity": [
            r"Research Maturity:\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"5\.\s*Research Maturity:\s*(\d+(?:\.\d+)?)\s*/\s*10"
        ]
    }
    
    scores_found = 0
    for dimension, patterns in score_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                scoring_data[dimension] = score
                scores_found += 1
                print(f"   ✓ Found {dimension}: {score}/10")
                
                # Store in dimension_details for structured output
                scoring_data["dimension_details"][dimension] = {
                    "score": score,
                    "weight": get_weight(dimension),
                    "evidence": "",
                    "analysis": ""
                }
                
                # Try to extract evidence and analysis for this dimension
                # Look for text between this dimension and the next
                dimension_section = re.search(
                    rf"{re.escape(match.group(0))}(.*?)(?:\d+\.\s*[A-Z]|CALCULATION:|$)",
                    full_text,
                    re.DOTALL | re.IGNORECASE
                )
                
                if dimension_section:
                    section_text = dimension_section.group(1)
                    
                    # Extract Evidence
                    evidence_match = re.search(r"Evidence:\s*(.+?)(?=\n\s*Analysis:|$)", section_text, re.DOTALL)
                    if evidence_match:
                        evidence = evidence_match.group(1).strip()
                        scoring_data["dimension_details"][dimension]["evidence"] = evidence
                    
                    # Extract Analysis
                    analysis_match = re.search(r"Analysis:\s*(.+?)(?=\n\s*\d+\.|$)", section_text, re.DOTALL)
                    if analysis_match:
                        analysis = analysis_match.group(1).strip()
                        scoring_data["dimension_details"][dimension]["analysis"] = analysis
                
                break  # Found score for this dimension, move to next
    
    scoring_data["scores_found"] = scores_found >= 3
    print(f"   📊 Found {scores_found}/5 dimension scores")
    
    # ===== CALCULATE WEIGHTED SCORE =====
    if scores_found >= 3:
        calculated = 0.0
        weights = {
            "domain_deep_thinking": 0.25,
            "systems_thinking": 0.25,
            "production_mindset": 0.20,
            "learning_velocity": 0.20,
            "research_maturity": 0.10
        }
        
        for dimension, weight in weights.items():
            if scoring_data[dimension] is not None:
                calculated += scoring_data[dimension] * weight
        
        scoring_data["calculated_score"] = round(calculated, 2)
        print(f"   🧮 Calculated weighted score: {scoring_data['calculated_score']}/10")
    
    # ===== EXTRACT OVERALL SCORE =====
    overall_patterns = [
        r"OVERALL WEIGHTED SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"Overall.*?Score:\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"Total.*?Score:\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"\*\*OVERALL WEIGHTED SCORE:\s*(\d+(?:\.\d+)?)/10\*\*"
    ]
    
    for pattern in overall_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            scoring_data["overall_score"] = float(match.group(1))
            print(f"   ✓ Found overall score: {scoring_data['overall_score']}/10")
            break
    
    # Use calculated score if overall score not found
    if scoring_data["overall_score"] is None and scoring_data["calculated_score"] is not None:
        scoring_data["overall_score"] = scoring_data["calculated_score"]
        print(f"   ℹ️  Using calculated score as overall: {scoring_data['overall_score']}/10")
    
    # ===== EXTRACT RECOMMENDATION =====
    recommendation_patterns = [
        (r"✅\s*STRONG HIRE", "STRONG HIRE"),
        (r"✅\s*HIRE(?:\s*-|\s*—|\.|\s*$)", "HIRE"),
        (r"⚠️\s*MAYBE", "MAYBE"),
        (r"❌\s*NO HIRE", "NO HIRE"),
        (r"HIRING RECOMMENDATION:\s*\*\*.*?(STRONG HIRE|HIRE|MAYBE|NO HIRE)", None)
    ]
    
    for pattern, fixed_value in recommendation_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            if fixed_value:
                scoring_data["recommendation"] = fixed_value
            else:
                scoring_data["recommendation"] = match.group(1).upper()
            
            print(f"   ✓ Found recommendation: {scoring_data['recommendation']}")
            
            # Extract reasoning
            reasoning_pattern = rf"{re.escape(match.group(0))}.*?Reasoning:\s*(.+?)(?=\n\*\*|$)"
            reasoning_match = re.search(reasoning_pattern, full_text, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                scoring_data["recommendation_reasoning"] = reasoning_match.group(1).strip()
            
            break
    
    # ===== EXTRACT KEY STRENGTHS =====
    strengths_pattern = r"\*\*KEY STRENGTHS:\*\*(.+?)(?:\*\*AREAS FOR GROWTH:|\*\*MEMORABLE QUOTES:|Do you have|$)"
    strengths_match = re.search(strengths_pattern, full_text, re.DOTALL | re.IGNORECASE)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strengths = re.findall(r"[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)", strengths_text, re.DOTALL)
        scoring_data["strengths"] = [s.strip() for s in strengths if s.strip()]
        print(f"   ✓ Found {len(scoring_data['strengths'])} key strengths")
    
    # ===== EXTRACT AREAS FOR GROWTH =====
    growth_pattern = r"\*\*AREAS FOR GROWTH:\*\*(.+?)(?:\*\*MEMORABLE QUOTES:|Do you have|$)"
    growth_match = re.search(growth_pattern, full_text, re.DOTALL | re.IGNORECASE)
    if growth_match:
        growth_text = growth_match.group(1)
        growth_areas = re.findall(r"[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)", growth_text, re.DOTALL)
        scoring_data["growth_areas"] = [g.strip() for g in growth_areas if g.strip()]
        print(f"   ✓ Found {len(scoring_data['growth_areas'])} growth areas")
    
    # ===== EXTRACT MEMORABLE QUOTES =====
    quotes_pattern = r"\*\*MEMORABLE QUOTES FROM CANDIDATE:\*\*(.+?)(?:Do you have|$)"
    quotes_match = re.search(quotes_pattern, full_text, re.DOTALL | re.IGNORECASE)
    if quotes_match:
        quotes_text = quotes_match.group(1)
        quotes = re.findall(r'"([^"]+)"', quotes_text)
        scoring_data["candidate_quotes"] = [q.strip() for q in quotes if q.strip()]
        print(f"   ✓ Found {len(scoring_data['candidate_quotes'])} memorable quotes")
    
    # Store evidence from old format for backward compatibility
    evidence_matches = re.findall(r"Evidence:\s*(.+?)(?=\n\s*Analysis:|\n\s*\d+\.)", full_text, re.DOTALL)
    if evidence_matches:
        scoring_data["evidence"] = {
            f"dimension_{i+1}": evidence.strip()
            for i, evidence in enumerate(evidence_matches[:5])
        }
    
    return scoring_data


def format_evaluation_summary(scoring_data: Dict) -> str:
    """
    Create a clean, readable summary like:
    - Domain-deep thinking: 9/10 (healthcare ML, understood domain first)
    - Systems thinking: 9/10 (hybrid rules + ML approach)
    - Total: 8.6/10 (Strong Hire)
    """
    
    if not scoring_data["scores_found"]:
        return "No evaluation found in transcript"
    
    lines = []
    
    dimension_labels = {
        "domain_deep_thinking": "Domain-deep thinking",
        "systems_thinking": "Systems thinking",
        "production_mindset": "Production mindset",
        "learning_velocity": "Learning velocity",
        "research_maturity": "Research maturity"
    }
    
    for key, label in dimension_labels.items():
        score = scoring_data.get(key)
        
        if score is not None:
            # Get insight from dimension_details if available
            insight = ""
            if key in scoring_data.get("dimension_details", {}):
                details = scoring_data["dimension_details"][key]
                evidence = details.get("evidence", "")
                if evidence:
                    insight = evidence[:60].strip()
                    if len(evidence) > 60:
                        insight += "..."
            
            if insight:
                lines.append(f"- {label}: {score}/10 ({insight})")
            else:
                lines.append(f"- {label}: {score}/10")
    
    # Add total score and recommendation
    total_score = scoring_data.get("overall_score") or scoring_data.get("calculated_score")
    recommendation = scoring_data.get("recommendation", "N/A")
    
    if total_score:
        lines.append(f"- **Total: {total_score}/10 ({recommendation})**")
    
    return "\n".join(lines)


@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Metrics storage
    all_metrics: List[Any] = []
    
    session = AgentSession(
        # Use inference API instead of direct plugin imports
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4o-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        all_metrics.append(ev.metrics)
        logger.info(f"Metrics collected: {type(ev.metrics).__name__}")

    session.on("metrics_collected", _on_metrics_collected)

    async def write_transcript():
        """
        Enhanced transcript writer with better scoring extraction and formatting.
        """
        try:
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = transcript_dir / f"interview_transcript_{ctx.room.name}_{current_date}.json"
            
            if not os.access(transcript_dir, os.W_OK):
                print(f"❌ NO WRITE PERMISSION: {transcript_dir}")
                print(f"💡 Run: chmod 755 {transcript_dir}")
                return
            
            history_dict = session.history.to_dict()
            
            # Extract what the candidate (user) said
            candidate_responses = extract_candidate_responses(history_dict)
            
            # Extract scoring that the agent provided based on candidate's responses
            scoring_data = extract_scoring_from_transcript(history_dict)
            
            # Compute latencies
            eou_delay = llm_ttft = tts_ttfb = 0.0
            metric_types = []
            
            for m in all_metrics:
                metric_types.append(type(m).__name__)
                if hasattr(m, 'end_of_utterance_delay'):
                    eou_delay = m.end_of_utterance_delay
                if hasattr(m, 'ttft') or hasattr(m, 'time_to_first_token'):
                    llm_ttft = getattr(m, 'ttft', getattr(m, 'time_to_first_token', 0.0))
                if hasattr(m, 'ttfb') or hasattr(m, 'time_to_first_byte'):
                    tts_ttfb = getattr(m, 'ttfb', getattr(m, 'time_to_first_byte', 0.0))
            
            total_latency = eou_delay + llm_ttft + tts_ttfb
            latency_data = {
                "end_of_utterance_delay": float(eou_delay),
                "llm_ttft": float(llm_ttft),
                "tts_ttfb": float(tts_ttfb),
                "total_latency": float(total_latency),
                "metrics_types": metric_types[-5:],
                "metrics_count": len(all_metrics)
            }
            
            # Build comprehensive output
            output_data = {
                "interview_metadata": {
                    "interview_type": "BaseThesis Technical Behavioral Interview",
                    "room_name": ctx.room.name,
                    "timestamp": current_date,
                    "interview_complete": scoring_data["scores_found"],
                    "candidate_response_count": len(candidate_responses)
                },
                
                "candidate_evaluation": {
                    "overall_score": scoring_data["overall_score"],
                    "calculated_score": scoring_data["calculated_score"],
                    "recommendation": scoring_data["recommendation"],
                    "recommendation_reasoning": scoring_data["recommendation_reasoning"],
                    "scores_found": scoring_data["scores_found"],
                    
                    # Individual dimension scores (flat structure for easy access)
                    "domain_deep_thinking": scoring_data["domain_deep_thinking"],
                    "systems_thinking": scoring_data["systems_thinking"],
                    "production_mindset": scoring_data["production_mindset"],
                    "learning_velocity": scoring_data["learning_velocity"],
                    "research_maturity": scoring_data["research_maturity"],
                    
                    # Detailed breakdown
                    "dimension_details": scoring_data["dimension_details"],
                    
                    # Qualitative feedback
                    "key_strengths": scoring_data["strengths"],
                    "areas_for_growth": scoring_data["growth_areas"],
                    "memorable_quotes": scoring_data["candidate_quotes"]
                },
                
                "candidate_responses": candidate_responses,
                
                "full_transcript": history_dict,
                
                "latency_metrics": latency_data,
                
                "evaluation_framework": {
                    "domain_deep_thinking": {"weight": "25%", "scale": "0-10"},
                    "systems_thinking": {"weight": "25%", "scale": "0-10"},
                    "production_mindset": {"weight": "20%", "scale": "0-10"},
                    "learning_velocity": {"weight": "20%", "scale": "0-10"},
                    "research_maturity": {"weight": "10%", "scale": "0-10"}
                }
            }
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                f.flush()
            
            # ===== CONSOLE OUTPUT =====
            print("\n" + "="*70)
            print(f"✅ INTERVIEW TRANSCRIPT SAVED")
            print("="*70)
            print(f"📁 Location: {filename.absolute()}")
            print(f"📊 Total latency: {total_latency:.3f}s ({len(all_metrics)} metrics)")
            print(f"💬 Candidate provided {len(candidate_responses)} responses")
            print()
            
            if scoring_data["scores_found"]:
                print("🎯 CANDIDATE EVALUATION SUMMARY")
                print("-"*70)
                print(format_evaluation_summary(scoring_data))
                print("-"*70)
                
                # Show individual scores
                if scoring_data["overall_score"]:
                    print(f"\n📈 Breakdown:")
                    print(f"   Domain-Deep: {scoring_data['domain_deep_thinking']}/10")
                    print(f"   Systems: {scoring_data['systems_thinking']}/10")
                    print(f"   Production: {scoring_data['production_mindset']}/10")
                    print(f"   Learning: {scoring_data['learning_velocity']}/10")
                    print(f"   Research: {scoring_data['research_maturity']}/10")
                
                # Additional details
                if scoring_data["strengths"]:
                    print(f"\n✨ Key Strengths: {len(scoring_data['strengths'])} identified")
                if scoring_data["growth_areas"]:
                    print(f"📈 Growth Areas: {len(scoring_data['growth_areas'])} identified")
                if scoring_data["candidate_quotes"]:
                    print(f"💭 Memorable Quotes: {len(scoring_data['candidate_quotes'])}")
                # Debug: show what we have
                if scoring_data["overall_score"]:
                   print(f"\n   Found overall score: {scoring_data['overall_score']}/10")
            
            print("="*70 + "\n")
        
        except PermissionError:
            print(f"❌ PERMISSION DENIED: {transcript_dir}")
            print(f"💡 Fix: chmod -R 755 {transcript_dir.parent}")
        except Exception as e:
            print(f"❌ SAVE ERROR: {e}")
            import traceback
            traceback.print_exc()

    ctx.add_shutdown_callback(write_transcript)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await ctx.connect()

    try:
        # Keep the session alive until it finishes
        await session.wait_for_completion()
    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        # Always save transcript when session ends
        await write_transcript()


if __name__ == "__main__":
    cli.run_app(server)