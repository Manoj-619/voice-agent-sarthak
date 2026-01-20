import logging
from datetime import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Any, Dict

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
os.environ["ULTRAVOX_API_KEY"] = "b7d3c841-055e-47c1-b3ce-09799bef8d02"

from livekit.plugins import noise_cancellation, silero, deepgram, cartesia, google, ultravox
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.ultravox.realtime import RealtimeModel


logger = logging.getLogger("agent")

load_dotenv(".env.local")

# YOUR EXACT PATH - Fixed and created at startup
TRANSCRIPT_PATH = "/Users/riapicardo/Desktop/basethesis/ottomator-agents/agent-starter-python/.github/transcripts1"
transcript_dir = Path(TRANSCRIPT_PATH)

# FORCE CREATE DIRECTORY AT STARTUP with full error reporting
try:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    print(f"Transcript directory ensured: {transcript_dir.absolute()}")
    print(f"Directory permissions: {'OK' if os.access(transcript_dir, os.W_OK) else 'FAILED'}")
except Exception as e:
    print(f"FAILED to create directory {TRANSCRIPT_PATH}")
    print(f"Error: {e}")
    # Fallback to home dir
    transcript_dir = Path.home() / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fallback directory: {transcript_dir.absolute()}")

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
- 8.0-10.0:  STRONG HIRE - Make offer immediately
  Reasoning: [Explain specific strengths from their answers]
  
- 6.0-7.9:  HIRE - Good fit, proceed with offer
  Reasoning: [Explain why they're a good fit despite not being perfect]
  
- 4.0-5.9:  MAYBE - Need more data points or different role fit
  Reasoning: [Explain what's missing and what additional info you'd need]
  
- 0.0-3.9:  NO HIRE - Not the right fit at this time
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

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Metrics 
   
    api_key = os.environ.get("ULTRAVOX_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"✅ ULTRAVOX API key verified: {len(api_key)} chars")
    # uv_model = RealtimeModel(
    #     model="fixie-ai/ultravox", # High-performance multimodal model
    #     voice="Mark",              # Professional male voice
    #     system_prompt=Assistant().instructions,
    #     temperature=0.8,
    #     first_speaker="FIRST_SPEAKER_AGENT" # Starts the interview immediately
    # )
    all_metrics: List[Any] = []
    
    session = AgentSession(
        # stt=deepgram.STT(model="nova-2-general"),
        # # llm=inference.LLM(model="openai/gpt-4o-mini"),
        # llm=ultravox.realtime.RealtimeModel(),
        # tts = inference.TTS(
        #     model="cartesia/sonic-3",
        #     voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        # ),
        # # tts=cartesia.TTS(
        # #     model="sonic-3",
        # #     voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"  # Professional male voice
        # # ),
        # turn_detection=MultilingualModel(),
        # vad=ctx.proc.userdata["vad"],
        # preemptive_generation=True,
        vad=ctx.proc.userdata["vad"],
      llm=RealtimeModel(
        model="ultravox-realtime",
        voice="Mark"),
    
    # llm=uv_model, 
    #     stt=None, 
    #     tts=None,
    #     turn_detection=uv_model, # Use Ultravox's built-in turn detection
    #     vad=ctx.proc.userdata["vad"],
    #     preemptive_generation=True,
    # stt=ultravox.STT(  # ✅ PASS API KEY EXPLICITLY
    #         api_key=api_key,
    #         model="ultravox-0.2"  # ✅ REQUIRED MODEL NAME
    #     ),
    #     llm=inference.LLM(model="openai/gpt-4o-mini"),
    #     tts=inference.TTS(
    #         model="cartesia/sonic-3",
    #         voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    #     ),
    #     turn_detection=MultilingualModel(),
    #     vad=ctx.proc.userdata["vad"],
    #     preemptive_generation=True,
        # vad=ctx.proc.userdata["vad"],
        # llm=RealtimeModel(
        #     model="fixie-ai/ultravox",
        #     voice="Mark",
        # ),
#         stt=ultravox(
#     api_key=api_key,
#     model="ultravox-0.2"
# ),
# llm=inference.LLM(model="openai/gpt-4o-mini"),
# tts=inference.TTS(
#     model="cartesia/sonic-3",
#     voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
# ),
# turn_detection=MultilingualModel(),
# vad=ctx.proc.userdata["vad"],


    )
    print("✅ AgentSession created with Ultravox STT")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        all_metrics.append(ev.metrics)
        logger.info(f"Metrics collected: {type(ev.metrics).__name__}")

    session.on("metrics_collected", _on_metrics_collected)

    async def write_transcript():
        try:
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = transcript_dir / f"interview_transcript_{ctx.room.name}_{current_date}.json"
            
            if not os.access(transcript_dir, os.W_OK):
                print(f"NO WRITE PERMISSION: {transcript_dir}")
                print(f"Run: chmod 755 {transcript_dir}")
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
            
            output_data = {
                "interview_type": "BaseThesis Technical Behavioral Interview",
                "transcript": history_dict,
                "candidate_responses": candidate_responses,
                "candidate_evaluation": scoring_data,
                "latency_metrics": latency_data,
                "room_name": ctx.room.name,
                "timestamp": current_date,
                "evaluation_framework": {
                    "domain_deep_thinking": {"weight": "25%", "scale": "0-10"},
                    "systems_thinking": {"weight": "25%", "scale": "0-10"},
                    "production_mindset": {"weight": "20%", "scale": "0-10"},
                    "learning_velocity": {"weight": "20%", "scale": "0-10"},
                    "research_maturity": {"weight": "10%", "scale": "0-10"}
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                f.flush()
            
            print(f" INTERVIEW SAVED: {filename.absolute()}")
            print(f" Total latency: {total_latency:.3f}s ({len(all_metrics)} metrics)")
            print(f" Candidate provided {len(candidate_responses)} responses")
            
            if scoring_data.get("overall_score"):
                print(f" Candidate Score: {scoring_data['overall_score']}/10 - {scoring_data.get('recommendation', 'N/A')}")
                print(f"   Domain-Deep: {scoring_data.get('domain_deep_thinking', 'N/A')}/10")
                print(f"   Systems: {scoring_data.get('systems_thinking', 'N/A')}/10")
                print(f"   Production: {scoring_data.get('production_mindset', 'N/A')}/10")
                print(f"   Learning: {scoring_data.get('learning_velocity', 'N/A')}/10")
                print(f"   Research: {scoring_data.get('research_maturity', 'N/A')}/10")
            else:
                print(f" Warning: No scoring data found - interview may have been incomplete")
            
        except PermissionError:
            print(f" PERMISSION DENIED: {transcript_dir}")
            print(f"Fix: chmod -R 755 {transcript_dir.parent}")
        except Exception as e:
            print(f" SAVE ERROR: {e}")
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
                    "content": content,
                    "confidence": item.get("transcript_confidence", 0)
                })
    
    return candidate_responses

def extract_scoring_from_transcript(history_dict: Dict) -> Dict:
    """
    Extract scoring information that the agent provided based on analyzing
    the candidate's (user's) responses.
    """
    import re
    
    scoring_data = {
        "scores_found": False,
        "domain_deep_thinking": None,
        "systems_thinking": None,
        "production_mindset": None,
        "learning_velocity": None,
        "research_maturity": None,
        "overall_score": None,
        "recommendation": None,
        "evidence": {},
        "analysis": {},
        "strengths": [],
        "growth_areas": [],
        "candidate_quotes": []
    }
    
    # Get the last few assistant messages (scoring is at the end)
    items = history_dict.get("items", [])
    assistant_messages = [item for item in items if item.get("role") == "assistant"]
    
    if not assistant_messages:
        return scoring_data
    
    # Check the last 5 assistant messages for scoring information
    last_messages = assistant_messages[-5:]
    full_text = " ".join([" ".join(msg.get("content", [])) for msg in last_messages])
    
    # Extract individual dimension scores
    score_patterns = {
        "domain_deep_thinking": r"Domain-Deep Thinking:\s*(\d+(?:\.\d+)?)/10",
        "systems_thinking": r"Systems Thinking:\s*(\d+(?:\.\d+)?)/10",
        "production_mindset": r"Production Mindset:\s*(\d+(?:\.\d+)?)/10",
        "learning_velocity": r"Learning Velocity:\s*(\d+(?:\.\d+)?)/10",
        "research_maturity": r"Research Maturity:\s*(\d+(?:\.\d+)?)/10"
    }
    
    for key, pattern in score_patterns.items():
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            scoring_data[key] = float(match.group(1))
            scoring_data["scores_found"] = True
    
    # Extract overall score
    overall_match = re.search(r"OVERALL WEIGHTED SCORE:\s*(\d+(?:\.\d+)?)/10", full_text, re.IGNORECASE)
    if overall_match:
        scoring_data["overall_score"] = float(overall_match.group(1))
    
    # Extract recommendation
    recommendation_patterns = [
        r"\s*STRONG HIRE",
        r"\s*HIRE",
        r"\s*MAYBE",
        r"\s*NO HIRE",
        r"(STRONG HIRE|HIRE|MAYBE|NO HIRE)"
    ]
    for pattern in recommendation_patterns:
        rec_match = re.search(pattern, full_text, re.IGNORECASE)
        if rec_match:
            rec_text = rec_match.group(0)
            if "STRONG HIRE" in rec_text.upper():
                scoring_data["recommendation"] = "STRONG HIRE"
            elif "NO HIRE" in rec_text.upper():
                scoring_data["recommendation"] = "NO HIRE"
            elif "HIRE" in rec_text.upper():
                scoring_data["recommendation"] = "HIRE"
            elif "MAYBE" in rec_text.upper():
                scoring_data["recommendation"] = "MAYBE"
            break
    
    # Extract evidence sections (what the candidate said)
    evidence_pattern = r"Evidence:\s*([^\n]+)"
    evidence_matches = re.findall(evidence_pattern, full_text)
    if evidence_matches:
        scoring_data["evidence"] = {
            f"dimension_{i+1}": evidence
            for i, evidence in enumerate(evidence_matches[:5])
        }
    
    # Extract strengths
    strengths_section = re.search(r"KEY STRENGTHS:(.+?)(?:AREAS FOR GROWTH:|MEMORABLE QUOTES:|$)", full_text, re.DOTALL | re.IGNORECASE)
    if strengths_section:
        strengths = re.findall(r"[-•]\s*(.+)", strengths_section.group(1))
        scoring_data["strengths"] = [s.strip() for s in strengths]
    
    # Extract growth areas
    growth_section = re.search(r"AREAS FOR GROWTH:(.+?)(?:MEMORABLE QUOTES:|$)", full_text, re.DOTALL | re.IGNORECASE)
    if growth_section:
        growth_areas = re.findall(r"[-•]\s*(.+)", growth_section.group(1))
        scoring_data["growth_areas"] = [g.strip() for g in growth_areas]
    
    # Extract candidate quotes
    quotes = re.findall(r'"([^"]+)"', full_text)
    if quotes:
        scoring_data["candidate_quotes"] = quotes[:5]  # Top 5 quotes
    
    return scoring_data

if __name__ == "__main__":
    cli.run_app(server)


# import logging
# from datetime import datetime
# import json
# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from typing import List, Any, Dict
# import asyncio
# import wave
# import io
# from livekit import rtc
# from livekit.agents import (
#     Agent,
#     AgentServer,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     cli,
#     inference,
#     room_io,
#     metrics,
#     MetricsCollectedEvent,
# )
# from livekit.plugins import noise_cancellation, silero, deepgram, cartesia, google
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("agent")

# load_dotenv(".env.local")

# # YOUR EXACT PATH - Fixed and created at startup
# TRANSCRIPT_PATH = "/Users/riapicardo/Desktop/basethesis/ottomator-agents/agent-starter-python/.github/transcripts1"
# transcript_dir = Path(TRANSCRIPT_PATH)

# # FORCE CREATE DIRECTORY AT STARTUP with full error reporting
# try:
#     transcript_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Transcript directory ensured: {transcript_dir.absolute()}")
#     print(f"Directory permissions: {'OK' if os.access(transcript_dir, os.W_OK) else 'FAILED'}")
# except Exception as e:
#     print(f"FAILED to create directory {TRANSCRIPT_PATH}")
#     print(f"Error: {e}")
#     # Fallback to home dir
#     transcript_dir = Path.home() / "transcripts"
#     transcript_dir.mkdir(parents=True, exist_ok=True)
#     print(f"Fallback directory: {transcript_dir.absolute()}")

# class AudioRecorder:
#     """Simple audio recorder for LiveKit room audio."""
#     def __init__(self, audio_path: Path):
#         self.audio_path = audio_path
#         self.frames: List[bytes] = []
#         self.sample_rate = 16000
#         self.channels = 1
#         self.sample_width = 2  # 16-bit
        
#     async def record_from_track(self, track: rtc.LocalAudioTrack):
#         """Record audio frames from a track."""
#         try:
#             async for frame in track:
#                 if frame.data:
#                     self.frames.append(frame.data)
#         except Exception as e:
#             print(f"Audio recording error: {e}")
    
#     def save_wav(self):
#         """Save collected frames as WAV file."""
#         if not self.frames:
#             print("No audio frames to save")
#             return False
            
#         try:
#             with wave.open(str(self.audio_path), 'wb') as wav_file:
#                 wav_file.setnchannels(self.channels)
#                 wav_file.setsampwidth(self.sample_width)
#                 wav_file.setframerate(self.sample_rate)
#                 wav_file.writeframes(b''.join(self.frames))
#             print(f"Audio saved: {self.audio_path.absolute()}")
#             return True
#         except Exception as e:
#             print(f"WAV save error: {e}")
#             return False

# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a professional technical interviewer for BaseThesis, conducting a behavioral interview to evaluate candidates across 5 key dimensions.

# YOUR MISSION: Assess learning velocity and systems thinking based on what the CANDIDATE says, not credentials or years of experience.

#    INTERVIEW STRUCTURE (5-7 minutes total):

#    1. INTRODUCTION (30 seconds):
#       - Warmly introduce yourself as a BaseThesis technical interviewer
#       - Briefly explain: "I'll ask you a few questions about your technical experiences to understand how you approach problems"
   
#    2. DOMAIN-DEEP THINKING (90 seconds):
#       Primary: "Tell me about a difficult technical problem you solved recently. What made it challenging?"
#       Follow-ups to ask based on response:
#       - "What alternatives did you consider? Why did you choose this approach?"
#       - "What didn't you know at the start, and how did you learn it?"
#       - "Looking back, what would you do differently?"
   
#       SCORING SIGNALS (based on what USER says):
#       - Strong (8-10): Explains WHY not just WHAT, references fundamentals, articulates unknowns, evidence of deep research
#       - Weak (0-4): Only describes WHAT was built, copies patterns, can't explain alternatives, surface-level understanding

#    3. SYSTEMS THINKING (90 seconds):
#       Primary: "Describe a time you made a technical decision that involved tradeoffs."
#       Follow-ups:
#       - "What could go wrong with that approach?"
#       - "How would you monitor it in production?"
#       - "Did you consider error handling or edge cases?"
   
#       SCORING SIGNALS (based on what USER says):
#       - Strong (8-10): Considers errors/edge cases unprompted, discusses tradeoffs, thinks about production concerns, hybrid approaches
#       - Weak (0-4): Only happy path thinking, optimizes one dimension, never mentions failures, pure solutions without tradeoffs

#    4. LEARNING VELOCITY & PRODUCTION MINDSET (90 seconds):
#       Primary: "Tell me about a time you had to learn something completely new and ship it."
#       Follow-ups:
#       - "What was your learning process? How did you know you understood it?"
#       - "Did anyone actually use what you built? What broke?"
#       - "How did you debug issues?"
   
#       SCORING SIGNALS FOR LEARNING (based on what USER says):
#       - Strong (8-10): Clear learning process, rapid skill acquisition, learns from failures, curious about WHY
#       - Weak (0-4): Relies on tutorials, doesn't adapt, stops at "it works"
   
#       SCORING SIGNALS FOR PRODUCTION (based on what USER says):
#       - Strong (8-10): Discusses error handling/logging/testing unprompted, evidence of shipping, mentions user impact
#       - Weak (0-4): Only features, no reliability talk, no evidence of real usage, optimizes for cleverness

#    5. RESEARCH MATURITY (60 seconds):
#       Primary: "When you approached that problem, what alternatives did you explore?"
#       Follow-ups:
#       - "Why didn't you use [alternative approach they mention]?"
#       - "If you had 10x more resources, what would you explore?"
   
#       SCORING SIGNALS (based on what USER says):
#       - Strong (8-10): Multiple approaches considered, knows why NOT chosen, reads papers/docs first, understands state of art
#       - Weak (0-4): Only one way, can't explain tradeoffs, no exploration, unaware of possibilities

#    6. CLOSING (30 seconds):
#       - Thank them professionally
#       - "Thank you for sharing your experiences with me today. Let me now analyze your responses and provide you with detailed feedback and scores."

#    7. **MANDATORY SCORING SECTION - YOU MUST PROVIDE THIS**:

#    After completing the interview questions, you MUST analyze what the CANDIDATE (user) said and provide a detailed evaluation in this EXACT format:

#    "Based on your responses during our conversation, here is your detailed evaluation:

#    **DIMENSION SCORES:**

#    1. Domain-Deep Thinking: [X]/10 (Weight: 25%)
#       Evidence: [Quote exactly what the candidate said that influenced this score]
#       Analysis: [Why this score - did they explain WHY or just WHAT? Did they mention fundamentals?]
   
#    2. Systems Thinking: [X]/10 (Weight: 25%)
#       Evidence: [Quote what the candidate said about tradeoffs, error handling, or edge cases]
#       Analysis: [Why this score - did they think beyond happy path?]
   
#    3. Production Mindset: [X]/10 (Weight: 20%)
#       Evidence: [Quote what the candidate said about shipping, debugging, or real usage]
#       Analysis: [Why this score - did they mention reliability, users, monitoring?]
   
#    4. Learning Velocity: [X]/10 (Weight: 20%)
#       Evidence: [Quote what the candidate said about their learning process]
#       Analysis: [Why this score - clear process? Rapid acquisition? Learn from failures?]
   
#    5. Research Maturity: [X]/10 (Weight: 10%)
#       Evidence: [Quote what the candidate said about alternatives they considered]
#       Analysis: [Why this score - multiple approaches? Knew why NOT to use alternatives?]

#    **CALCULATION:**
#    Domain-Deep Thinking: [X] × 0.25 = [Y]
#    Systems Thinking: [X] × 0.25 = [Y]
#    Production Mindset: [X] × 0.20 = [Y]
#    Learning Velocity: [X] × 0.20 = [Y]
#    Research Maturity: [X] × 0.10 = [Y]

#    **OVERALL WEIGHTED SCORE: [X.X]/10**

#    **HIRING RECOMMENDATION:**
#    [Choose based on total score:]
#    - 8.0-10.0:  STRONG HIRE - Make offer immediately
#      Reasoning: [Explain specific strengths from their answers]
  
#    - 6.0-7.9:  HIRE - Good fit, proceed with offer
#      Reasoning: [Explain why they're a good fit despite not being perfect]
  
#    - 4.0-5.9:  MAYBE - Need more data points or different role fit
#      Reasoning: [Explain what's missing and what additional info you'd need]
  
#    - 0.0-3.9:  NO HIRE - Not the right fit at this time
#      Reasoning: [Explain constructively what gaps exist]

#    **KEY STRENGTHS:**
#    - [Strength 1 with specific quote from candidate]
#    - [Strength 2 with specific quote from candidate]

#    **AREAS FOR GROWTH:**
#    - [Area 1 with specific example from their responses]
#    - [Area 2 with specific example from their responses]

#    **MEMORABLE QUOTES FROM CANDIDATE:**
#    - "[Exact quote that shows strong thinking]"
#    - "[Exact quote that influenced scoring]"

#    Do you have any questions about this evaluation?"

#    CRITICAL INSTRUCTIONS:
#    - You MUST complete all interview questions AND provide the full scoring section above
#    - Base ALL scores on what the CANDIDATE (user) actually said, not what you think they should have said
#    - Quote the candidate's actual words as evidence
#    - Be honest in scoring - if they gave weak answers, score accordingly
#    - If candidate gives very brief or unclear answers, probe deeper before scoring low
#    - ALWAYS provide complete numerical scores and hiring recommendation
#    - Show your calculation clearly so it's transparent

#    CONVERSATIONAL STYLE:
#    - Keep it warm and professional, not robotic
#    - Ask follow-ups naturally based on their responses
#    - If answers are too brief, encourage them: "Can you tell me more about that?"
#    - Don't rush - let them think and elaborate
#    - Maintain 5-7 minute total duration
#    - Be encouraging even if answers are weak
#    - ALWAYS provide complete scoring at the end

#  REMEMBER: You're evaluating HOW they think based on WHAT THEY SAY, not what they've built. Look for evidence of learning velocity, systems thinking, and production mindset in their actual words.
# """,
#         )

# server = AgentServer()

# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()

# server.setup_fnc = prewarm

# @server.rtc_session()
# async def my_agent(ctx: JobContext):
#     ctx.log_context_fields = {"room": ctx.room.name}

#     # Metrics storage
#     all_metrics: List[Any] = []
    
#     # Audio recording setup
#     current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
#     audio_file = transcript_dir / f"interview_audio_{ctx.room.name}_{current_date}.wav"
#     audio_recorder = AudioRecorder(audio_file)
#     audio_recording_started_at = None
    
#     session = AgentSession(
#         stt=deepgram.STT(model="nova-2-general"),
#         llm=inference.LLM(model="openai/gpt-4o-mini"),
#         tts=inference.TTS(
#             model="cartesia/sonic-3",
#             voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
#         ),
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         preemptive_generation=True,
#     )

#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         all_metrics.append(ev.metrics)
#         logger.info(f"Metrics collected: {type(ev.metrics).__name__}")

#     session.on("metrics_collected", _on_metrics_collected)

#     # Audio recording event handlers
#     @ctx.room.on("track_subscribed")
#     def on_track_subscribed(
#         track: rtc.LocalAudioTrack, 
#         publication: rtc.RemoteTrackPublication, 
#         participant: rtc.RemoteParticipant
#     ):
#         if track.kind == rtc.TrackKind.KIND_AUDIO and participant.identity != "agent":
#             print(f"Started recording audio from participant: {participant.identity}")
#             # Start recording task
#             asyncio.create_task(audio_recorder.record_from_track(track))
#             nonlocal audio_recording_started_at
#             if audio_recording_started_at is None:
#                 audio_recording_started_at = datetime.now().isoformat()

#     async def write_transcript():
#         try:
#             filename = transcript_dir / f"interview_transcript_{ctx.room.name}_{current_date}.json"
            
#             if not os.access(transcript_dir, os.W_OK):
#                 print(f"NO WRITE PERMISSION: {transcript_dir}")
#                 print(f"Run: chmod 755 {transcript_dir}")
#                 return
            
#             # Save audio first
#             audio_saved = audio_recorder.save_wav()
            
#             history_dict = session.history.to_dict()
            
#             # Extract what the candidate (user) said
#             candidate_responses = extract_candidate_responses(history_dict)
            
#             # Extract scoring that the agent provided based on candidate's responses
#             scoring_data = extract_scoring_from_transcript(history_dict)
            
#             # Compute latencies
#             eou_delay = llm_ttft = tts_ttfb = 0.0
#             metric_types = []
            
#             for m in all_metrics:
#                 metric_types.append(type(m).__name__)
#                 if hasattr(m, 'end_of_utterance_delay'):
#                     eou_delay = m.end_of_utterance_delay
#                 if hasattr(m, 'ttft') or hasattr(m, 'time_to_first_token'):
#                     llm_ttft = getattr(m, 'ttft', getattr(m, 'time_to_first_token', 0.0))
#                 if hasattr(m, 'ttfb') or hasattr(m, 'time_to_first_byte'):
#                     tts_ttfb = getattr(m, 'ttfb', getattr(m, 'time_to_first_byte', 0.0))
            
#             total_latency = eou_delay + llm_ttft + tts_ttfb
#             latency_data = {
#                 "end_of_utterance_delay": float(eou_delay),
#                 "llm_ttft": float(llm_ttft),
#                 "tts_ttfb": float(tts_ttfb),
#                 "total_latency": float(total_latency),
#                 "metrics_types": metric_types[-5:],
#                 "metrics_count": len(all_metrics)
#             }
            
#             output_data = {
#                 "interview_type": "BaseThesis Technical Behavioral Interview",
#                 "transcript": history_dict,
#                 "candidate_responses": candidate_responses,
#                 "candidate_evaluation": scoring_data,
#                 "latency_metrics": latency_data,
#                 "room_name": ctx.room.name,
#                 "timestamp": current_date,
#                 "audio_recording_path": str(audio_file) if audio_saved else None,
#                 "audio_recording_started_at": audio_recording_started_at,
#                 "evaluation_framework": {
#                     "domain_deep_thinking": {"weight": "25%", "scale": "0-10"},
#                     "systems_thinking": {"weight": "25%", "scale": "0-10"},
#                     "production_mindset": {"weight": "20%", "scale": "0-10"},
#                     "learning_velocity": {"weight": "20%", "scale": "0-10"},
#                     "research_maturity": {"weight": "10%", "scale": "0-10"}
#                 }
#             }
            
#             with open(filename, 'w', encoding='utf-8') as f:
#                 json.dump(output_data, f, indent=2, ensure_ascii=False)
#                 f.flush()
            
#             print(f" INTERVIEW SAVED: {filename.absolute()}")
#             if audio_saved:
#                 print(f" AUDIO SAVED: {audio_file.absolute()}")
#             print(f" Total latency: {total_latency:.3f}s ({len(all_metrics)} metrics)")
#             print(f" Candidate provided {len(candidate_responses)} responses")
            
#             if scoring_data.get("overall_score"):
#                 print(f" Candidate Score: {scoring_data['overall_score']}/10 - {scoring_data.get('recommendation', 'N/A')}")
#                 print(f"   Domain-Deep: {scoring_data.get('domain_deep_thinking', 'N/A')}/10")
#                 print(f"   Systems: {scoring_data.get('systems_thinking', 'N/A')}/10")
#                 print(f"   Production: {scoring_data.get('production_mindset', 'N/A')}/10")
#                 print(f"   Learning: {scoring_data.get('learning_velocity', 'N/A')}/10")
#                 print(f"   Research: {scoring_data.get('research_maturity', 'N/A')}/10")
#             else:
#                 print(f" Warning: No scoring data found - interview may have been incomplete")
            
#         except PermissionError:
#             print(f" PERMISSION DENIED: {transcript_dir}")
#             print(f"Fix: chmod -R 755 {transcript_dir.parent}")
#         except Exception as e:
#             print(f" SAVE ERROR: {e}")
#             import traceback
#             traceback.print_exc()

#     ctx.add_shutdown_callback(write_transcript)

#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=lambda params: (
#                     noise_cancellation.BVCTelephony()
#                     if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
#                     else noise_cancellation.BVC()
#                 ),
#             ),
#         ),
#     )

#     await ctx.connect()

# # Keep your existing extract_candidate_responses and extract_scoring_from_transcript functions unchanged
# def extract_candidate_responses(history_dict: Dict) -> List[Dict]:
#     """Extract all responses from the candidate (user messages)."""
#     candidate_responses = []
    
#     items = history_dict.get("items", [])
    
#     for item in items:
#         if item.get("role") == "user":
#             content = " ".join(item.get("content", []))
#             if content and content.strip():  # Only include non-empty responses
#                 candidate_responses.append({
#                     "timestamp": item.get("metrics", {}).get("started_speaking_at"),
#                     "content": content,
#                     "confidence": item.get("transcript_confidence", 0)
#                 })
    
#     return candidate_responses

# def extract_scoring_from_transcript(history_dict: Dict) -> Dict:
#     """Extract scoring information that the agent provided based on analyzing the candidate's (user's) responses."""
#     import re
    
#     scoring_data = {
#         "scores_found": False,
#         "domain_deep_thinking": None,
#         "systems_thinking": None,
#         "production_mindset": None,
#         "learning_velocity": None,
#         "research_maturity": None,
#         "overall_score": None,
#         "recommendation": None,
#         "evidence": {},
#         "analysis": {},
#         "strengths": [],
#         "growth_areas": [],
#         "candidate_quotes": []
#     }
    
#     # Get the last few assistant messages (scoring is at the end)
#     items = history_dict.get("items", [])
#     assistant_messages = [item for item in items if item.get("role") == "assistant"]
    
#     if not assistant_messages:
#         return scoring_data
    
#     # Check the last 5 assistant messages for scoring information
#     last_messages = assistant_messages[-5:]
#     full_text = " ".join([" ".join(msg.get("content", [])) for msg in last_messages])
    
#     # Extract individual dimension scores
#     score_patterns = {
#         "domain_deep_thinking": r"Domain-Deep Thinking:\s*(\d+(?:\.\d+)?)/10",
#         "systems_thinking": r"Systems Thinking:\s*(\d+(?:\.\d+)?)/10",
#         "production_mindset": r"Production Mindset:\s*(\d+(?:\.\d+)?)/10",
#         "learning_velocity": r"Learning Velocity:\s*(\d+(?:\.\d+)?)/10",
#         "research_maturity": r"Research Maturity:\s*(\d+(?:\.\d+)?)/10"
#     }
    
#     for key, pattern in score_patterns.items():
#         match = re.search(pattern, full_text, re.IGNORECASE)
#         if match:
#             scoring_data[key] = float(match.group(1))
#             scoring_data["scores_found"] = True
    
#     # Extract overall score
#     overall_match = re.search(r"OVERALL WEIGHTED SCORE:\s*(\d+(?:\.\d+)?)/10", full_text, re.IGNORECASE)
#     if overall_match:
#         scoring_data["overall_score"] = float(overall_match.group(1))
    
#     # Extract recommendation
#     recommendation_patterns = [
#         r"\s*STRONG HIRE",
#         r"\s*HIRE",
#         r"\s*MAYBE",
#         r"\s*NO HIRE",
#         r"(STRONG HIRE|HIRE|MAYBE|NO HIRE)"
#     ]
#     for pattern in recommendation_patterns:
#         rec_match = re.search(pattern, full_text, re.IGNORECASE)
#         if rec_match:
#             rec_text = rec_match.group(0)
#             if "STRONG HIRE" in rec_text.upper():
#                 scoring_data["recommendation"] = "STRONG HIRE"
#             elif "NO HIRE" in rec_text.upper():
#                 scoring_data["recommendation"] = "NO HIRE"
#             elif "HIRE" in rec_text.upper():
#                 scoring_data["recommendation"] = "HIRE"
#             elif "MAYBE" in rec_text.upper():
#                 scoring_data["recommendation"] = "MAYBE"
#             break
    
#     # Extract evidence sections (what the candidate said)
#     evidence_pattern = r"Evidence:\s*([^\n]+)"
#     evidence_matches = re.findall(evidence_pattern, full_text)
#     if evidence_matches:
#         scoring_data["evidence"] = {
#             f"dimension_{i+1}": evidence
#             for i, evidence in enumerate(evidence_matches[:5])
#         }
    
#     # Extract strengths
#     strengths_section = re.search(r"KEY STRENGTHS:(.+?)(?:AREAS FOR GROWTH:|MEMORABLE QUOTES:|$)", full_text, re.DOTALL | re.IGNORECASE)
#     if strengths_section:
#         strengths = re.findall(r"[-•]\s*(.+)", strengths_section.group(1))
#         scoring_data["strengths"] = [s.strip() for s in strengths]
    
#     # Extract growth areas
#     growth_section = re.search(r"AREAS FOR GROWTH:(.+?)(?:MEMORABLE QUOTES:|$)", full_text, re.DOTALL | re.IGNORECASE)
#     if growth_section:
#         growth_areas = re.findall(r"[-•]\s*(.+)", growth_section.group(1))
#         scoring_data["growth_areas"] = [g.strip() for g in growth_areas]
    
#     # Extract candidate quotes
#     quotes = re.findall(r'"([^"]+)"', full_text)
#     if quotes:
#         scoring_data["candidate_quotes"] = quotes[:5]  # Top 5 quotes
    
#     return scoring_data

# if __name__ == "__main__":
#     cli.run_app(server)


# import logging
# from datetime import datetime
# import json
# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from typing import List, Any, Dict
# import asyncio
# import wave
# import numpy as np
# from livekit import rtc
# from livekit.agents import (
#     Agent,
#     AgentServer,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     cli,
#     inference,
#     room_io,
#     metrics,
#     MetricsCollectedEvent,
# )
# from livekit.plugins import noise_cancellation, silero, deepgram, cartesia
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("agent")
# load_dotenv(".env.local")

# # 🔥 EXPLICIT PATH - CHANGE THIS TO YOUR DESIRED LOCATION
# SAVE_PATH = Path("/Users/riapicardo/Desktop/basethesis/ottomator-agents/agent-starter-python/.github/transcripts1")
# SAVE_PATH.mkdir(parents=True, exist_ok=True)

# print(f"🚀 FILES WILL BE SAVED HERE: {SAVE_PATH.absolute()}")
# print(f"📁 Directory writable: {'✅ YES' if os.access(SAVE_PATH, os.W_OK) else '❌ NO - FIX PERMISSIONS'}")

# class SimpleAudioRecorder:
#     def __init__(self, base_path: Path):
#         self.base_path = base_path
#         self.audio_frames = []
#         self.sample_rate = 16000
#         self.channels = 1
#         self.sample_width = 2
        
#     def add_frame(self, frame_data: bytes):
#         if frame_data and len(frame_data) > 0:
#             self.audio_frames.append(frame_data)
    
#     def save_wav(self, room_name: str, timestamp: str) -> Path:
#         if not self.audio_frames:
#             print("⚠️  No audio frames to save")
#             return None
        
#         audio_filename = f"🎤_AUDIO_{room_name}_{timestamp}.wav"
#         audio_path = self.base_path / audio_filename
        
#         print(f"\n💾 SAVING AUDIO TO: {audio_path.absolute()}")
        
#         try:
#             raw_data = b''.join(self.audio_frames)
#             with wave.open(str(audio_path), 'wb') as wav_file:
#                 wav_file.setnchannels(self.channels)
#                 wav_file.setsampwidth(self.sample_width)
#                 wav_file.setframerate(self.sample_rate)
#                 wav_file.writeframes(raw_data)
            
#             file_size = audio_path.stat().st_size / (1024*1024)  # MB
#             print(f"✅ AUDIO SAVED! 📊 Size: {file_size:.1f}MB, Frames: {len(self.audio_frames)}")
#             self.audio_frames.clear()
#             return audio_path
            
#         except Exception as e:
#             print(f"❌ AUDIO SAVE FAILED: {e}")
#             return None

# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a professional technical interviewer for BaseThesis, conducting a behavioral interview to evaluate candidates across 5 key dimensions.

# YOUR MISSION: Assess learning velocity and systems thinking based on what the CANDIDATE says, not credentials or years of experience.

#    INTERVIEW STRUCTURE (5-7 minutes total):

#    1. INTRODUCTION (30 seconds):
#       - Warmly introduce yourself as a BaseThesis technical interviewer
#       - Briefly explain: "I'll ask you a few questions about your technical experiences to understand how you approach problems"
   
#    2. DOMAIN-DEEP THINKING (90 seconds):
#       Primary: "Tell me about a difficult technical problem you solved recently. What made it challenging?"
#       Follow-ups to ask based on response:
#       - "What alternatives did you consider? Why did you choose this approach?"
#       - "What didn't you know at the start, and how did you learn it?"
#       - "Looking back, what would you do differently?"
   
#       SCORING SIGNALS (based on what USER says):
#       - Strong (8-10): Explains WHY not just WHAT, references fundamentals, articulates unknowns, evidence of deep research
#       - Weak (0-4): Only describes WHAT was built, copies patterns, can't explain alternatives, surface-level understanding

#    3. SYSTEMS THINKING (90 seconds):
#       Primary: "Describe a time you made a technical decision that involved tradeoffs."
#       Follow-ups:
#       - "What could go wrong with that approach?"
#       - "How would you monitor it in production?"
#       - "Did you consider error handling or edge cases?"
   
#       SCORING SIGNALS (based on what USER says):
#       - Strong (8-10): Considers errors/edge cases unprompted, discusses tradeoffs, thinks about production concerns, hybrid approaches
#       - Weak (0-4): Only happy path thinking, optimizes one dimension, never mentions failures, pure solutions without tradeoffs

#    4. LEARNING VELOCITY & PRODUCTION MINDSET (90 seconds):
#       Primary: "Tell me about a time you had to learn something completely new and ship it."
#       Follow-ups:
#       - "What was your learning process? How did you know you understood it?"
#       - "Did anyone actually use what you built? What broke?"
#       - "How did you debug issues?"
   
#       SCORING SIGNALS FOR LEARNING (based on what USER says):
#       - Strong (8-10): Clear learning process, rapid skill acquisition, learns from failures, curious about WHY
#       - Weak (0-4): Relies on tutorials, doesn't adapt, stops at "it works"
   
#       SCORING SIGNALS FOR PRODUCTION (based on what USER says):
#       - Strong (8-10): Discusses error handling/logging/testing unprompted, evidence of shipping, mentions user impact
#       - Weak (0-4): Only features, no reliability talk, no evidence of real usage, optimizes for cleverness

#    5. RESEARCH MATURITY (60 seconds):
#       Primary: "When you approached that problem, what alternatives did you explore?"
#       Follow-ups:
#       - "Why didn't you use [alternative approach they mention]?"
#       - "If you had 10x more resources, what would you explore?"
   
#       SCORING SIGNALS (based on what USER says):
#       - Strong (8-10): Multiple approaches considered, knows why NOT chosen, reads papers/docs first, understands state of art
#       - Weak (0-4): Only one way, can't explain tradeoffs, no exploration, unaware of possibilities

#    6. CLOSING (30 seconds):
#       - Thank them professionally
#       - "Thank you for sharing your experiences with me today. Let me now analyze your responses and provide you with detailed feedback and scores."

#    7. **MANDATORY SCORING SECTION - YOU MUST PROVIDE THIS**:

#    After completing the interview questions, you MUST analyze what the CANDIDATE (user) said and provide a detailed evaluation in this EXACT format:

#    "Based on your responses during our conversation, here is your detailed evaluation:

#    **DIMENSION SCORES:**

#    1. Domain-Deep Thinking: [X]/10 (Weight: 25%)
#       Evidence: [Quote exactly what the candidate said that influenced this score]
#       Analysis: [Why this score - did they explain WHY or just WHAT? Did they mention fundamentals?]
   
#    2. Systems Thinking: [X]/10 (Weight: 25%)
#       Evidence: [Quote what the candidate said about tradeoffs, error handling, or edge cases]
#       Analysis: [Why this score - did they think beyond happy path?]
   
#    3. Production Mindset: [X]/10 (Weight: 20%)
#       Evidence: [Quote what the candidate said about shipping, debugging, or real usage]
#       Analysis: [Why this score - did they mention reliability, users, monitoring?]
   
#    4. Learning Velocity: [X]/10 (Weight: 20%)
#       Evidence: [Quote what the candidate said about their learning process]
#       Analysis: [Why this score - clear process? Rapid acquisition? Learn from failures?]
   
#    5. Research Maturity: [X]/10 (Weight: 10%)
#       Evidence: [Quote what the candidate said about alternatives they considered]
#       Analysis: [Why this score - multiple approaches? Knew why NOT to use alternatives?]

#    **CALCULATION:**
#    Domain-Deep Thinking: [X] × 0.25 = [Y]
#    Systems Thinking: [X] × 0.25 = [Y]
#    Production Mindset: [X] × 0.20 = [Y]
#    Learning Velocity: [X] × 0.20 = [Y]
#    Research Maturity: [X] × 0.10 = [Y]

#    **OVERALL WEIGHTED SCORE: [X.X]/10**

#    **HIRING RECOMMENDATION:**
#    [Choose based on total score:]
#    - 8.0-10.0:  STRONG HIRE - Make offer immediately
#      Reasoning: [Explain specific strengths from their answers]
  
#    - 6.0-7.9:  HIRE - Good fit, proceed with offer
#      Reasoning: [Explain why they're a good fit despite not being perfect]
  
#    - 4.0-5.9:  MAYBE - Need more data points or different role fit
#      Reasoning: [Explain what's missing and what additional info you'd need]
  
#    - 0.0-3.9:  NO HIRE - Not the right fit at this time
#      Reasoning: [Explain constructively what gaps exist]

#    **KEY STRENGTHS:**
#    - [Strength 1 with specific quote from candidate]
#    - [Strength 2 with specific quote from candidate]

#    **AREAS FOR GROWTH:**
#    - [Area 1 with specific example from their responses]
#    - [Area 2 with specific example from their responses]

#    **MEMORABLE QUOTES FROM CANDIDATE:**
#    - "[Exact quote that shows strong thinking]"
#    - "[Exact quote that influenced scoring]"

#    Do you have any questions about this evaluation?"

#    CRITICAL INSTRUCTIONS:
#    - You MUST complete all interview questions AND provide the full scoring section above
#    - Base ALL scores on what the CANDIDATE (user) actually said, not what you think they should have said
#    - Quote the candidate's actual words as evidence
#    - Be honest in scoring - if they gave weak answers, score accordingly
#    - If candidate gives very brief or unclear answers, probe deeper before scoring low
#    - ALWAYS provide complete numerical scores and hiring recommendation
#    - Show your calculation clearly so it's transparent

#    CONVERSATIONAL STYLE:
#    - Keep it warm and professional, not robotic
#    - Ask follow-ups naturally based on their responses
#    - If answers are too brief, encourage them: "Can you tell me more about that?"
#    - Don't rush - let them think and elaborate
#    - Maintain 5-7 minute total duration
#    - Be encouraging even if answers are weak
#    - ALWAYS provide complete scoring at the end

#  REMEMBER: You're evaluating HOW they think based on WHAT THEY SAY, not what they've built. Look for evidence of learning velocity, systems thinking, and production mindset in their actual words.
#             """,
#         )

# server = AgentServer()

# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()
# server.setup_fnc = prewarm
# @server.rtc_session()
# async def my_agent(ctx: JobContext):
#     ctx.log_context_fields = {"room": ctx.room.name}
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     room_name = ctx.room.name.replace(" ", "_")
    
#     print(f"\n🎯 SESSION STARTED - Room: {room_name}")
#     print(f"📂 SAVING TO: {SAVE_PATH.absolute()}")
    
#     # 🎤 STT-BASED AUDIO CAPTURE (WORKS 100% IN CONSOLE-ROOM)
#     class AudioCaptureSTT(deepgram.STT):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.audio_frames = []
        
#         async def transcribe(self, audio_frame):
#             if audio_frame and audio_frame.data:
#                 self.audio_frames.append(audio_frame.data)
#             return await super().transcribe(audio_frame)
    
#     stt_with_recording = AudioCaptureSTT(model="nova-2-general")
    
#     session = AgentSession(
#         stt=stt_with_recording,
#         llm=inference.LLM(model="openai/gpt-4o-mini"),
#         tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         preemptive_generation=True,
#     )
    
#     all_metrics = []
#     session.on("metrics_collected", lambda ev: all_metrics.append(ev.metrics))
    
#     async def save_everything():
#         print(f"\n🔚 SAVING EVERYTHING...")
        
#         # 🎤 SAVE STT-CAPTURED AUDIO (THIS WILL WORK!)
#         audio_path = None
#         if stt_with_recording.audio_frames:
#             audio_filename = f"🎤_STT_AUDIO_{room_name}_{timestamp}.wav"
#             audio_path = SAVE_PATH / audio_filename
            
#             print(f"💾 SAVING {len(stt_with_recording.audio_frames)} audio frames...")
#             raw_data = b''.join(stt_with_recording.audio_frames)
            
#             with wave.open(str(audio_path), 'wb') as wav_file:
#                 wav_file.setnchannels(1)
#                 wav_file.setsampwidth(2)
#                 wav_file.setframerate(16000)
#                 wav_file.writeframes(raw_data)
            
#             size_mb = audio_path.stat().st_size / (1024*1024)
#             print(f"✅ AUDIO SAVED: {audio_path.absolute()} ({size_mb:.1f}MB)")
#         else:
#             print("⚠️  No STT audio frames captured")
        
#         # Save JSON (unchanged)
#         json_path = SAVE_PATH / f"📝_TRANSCRIPT_{room_name}_{timestamp}.json"
#         history_dict = session.history.to_dict()
#         candidate_responses = extract_candidate_responses(history_dict)
        
#         output_data = {
#             "save_paths": {
#                 "audio": str(audio_path) if audio_path else None,
#                 "json": str(json_path.absolute()),
#                 "directory": str(SAVE_PATH.absolute())
#             },
#             "audio_frames_captured": len(stt_with_recording.audio_frames),
#             "candidate_responses": candidate_responses,
#             "transcript": history_dict,
#             "room": room_name,
#             "timestamp": timestamp
#         }
        
#         with open(json_path, 'w') as f:
#             json.dump(output_data, f, indent=2)
        
#         print(f"🎉 COMPLETE! Audio: {'✅ YES' if audio_path else '❌ NO'} | JSON: {json_path.absolute()}")
    
#     ctx.add_shutdown_callback(save_everything)
    
#     await session.start(agent=Assistant(), room=ctx.room)
#     await ctx.connect()
# # Keep your existing extract functions (unchanged)
# def extract_candidate_responses(history_dict: Dict) -> List[Dict]:
#     candidate_responses = []
#     for item in history_dict.get("items", []):
#         if item.get("role") == "user":
#             content = " ".join(item.get("content", []))
#             if content.strip():
#                 candidate_responses.append({
#                     "content": content,
#                     "confidence": item.get("transcript_confidence", 0)
#                 })
#     return candidate_responses

# def extract_scoring_from_transcript(history_dict: Dict) -> Dict:
#     import re
#     scoring_data = {"scores_found": False}
#     assistant_messages = [item for item in history_dict.get("items", []) if item.get("role") == "assistant"]
#     if assistant_messages:
#         full_text = " ".join([" ".join(msg.get("content", [])) for msg in assistant_messages[-3:]])
#         for dim, pattern in {
#             "domain_deep_thinking": r"Domain-Deep Thinking:\s*(\d+(?:\.\d+)?)/10",
#             "systems_thinking": r"Systems Thinking:\s*(\d+(?:\.\d+)?)/10",
#         }.items():
#             match = re.search(pattern, full_text, re.IGNORECASE)
#             if match:
#                 scoring_data[dim] = float(match.group(1))
#                 scoring_data["scores_found"] = True
#     return scoring_data

# if __name__ == "__main__":
#     cli.run_app(server)

