import logging
import json
import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict
from dotenv import load_dotenv

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
from livekit.plugins import noise_cancellation, silero, deepgram, cartesia

# Load Environment Variables
load_dotenv(".env.local")
logger = logging.getLogger("agent")

# --- 1. CONFIGURATION & DATA ---

TRANSCRIPT_PATH = "/Users/riapicardo/Desktop/basethesis/ottomator-agents/agent-starter-python/.github/transcripts1"
transcript_dir = Path(TRANSCRIPT_PATH)
transcript_dir.mkdir(parents=True, exist_ok=True)

# The "Dummy Resume" used for context
DUMMY_RESUME = {
    "candidate_info": {
        "name": "Ria",
        "role": "AI Engineer",
        "key_project": "Healthcare RAG system for diagnostic assistance"
    },
    "tech_stack": ["Python", "Vector DBs", "LangChain", "RLHF"],
    "background": "Developed a hybrid rules + ML approach for high-stakes medical data."
}

# --- 2. THE INTERVIEWER (ASSISTANT) ---

class BaseThesisAssistant(Agent):
    def __init__(self, candidate: dict) -> None:
        instructions = f"""You are a senior technical interviewer at BaseThesis. 
You are interviewing {candidate['candidate_info']['name']} for an {candidate['candidate_info']['role']} position.

INTERVIEW GOAL:
Extract 'Strong Signals' for our hiring rubric: Domain Depth, Systems Thinking, Production Mindset, Learning Velocity, and Research Maturity.

CANDIDATE CONTEXT:
- Project: {candidate['candidate_info']['key_project']}
- Background: {candidate['background']}

INTERVIEW STRUCTURE (5-7 MINS):
1. Introduction: Briefly welcome them and reference {candidate['candidate_info']['key_project']}.
2. Domain Thinking: "Why did you choose a hybrid rules+ML approach? What were the fundamental constraints?"
3. Systems Thinking: "How does this system handle failure? What specific tradeoffs did you make for production reliability?"
4. Production Mindset: "Tell me about a time this system broke. How did you identify the root cause?"
5. Learning Velocity: "You used {candidate['tech_stack'][2]}. How did you master that quickly?"

RULES:
- Be rigorous. If answers are surface-level (WHAT), push for the WHY.
- If they mention a tool, ask why they didn't use an alternative (Research Maturity).
"""
        super().__init__(instructions=instructions)

# --- 3. THE SCORING & CONFIDENCE TOOL ---

async def generate_hiring_report(history: list, llm: inference.LLM):
    """Analyze transcript, calculate weighted scores, and assess confidence."""
    
    transcript_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    
    analysis_prompt = f"""
    You are an expert hiring auditor for BaseThesis. Based on the transcript, evaluate the candidate.

    1. CATEGORY SCORES (0-10):
       - Domain-Deep Thinking (25% weight)
       - Systems Thinking (25% weight)
       - Production Mindset (20% weight)
       - Learning Velocity (20% weight)
       - Research Maturity (10% weight)

    2. CONFIDENCE SCORE (0-10):
       - How much evidence was in the transcript? 
       - Low (0-4) if candidate was vague or interview was too short.
       - High (8-10) if specific technical details and tradeoffs were provided.

    Transcript:
    {transcript_text}

    Return ONLY a JSON object:
    {{
      "scores": {{
        "domain_thinking": 0,
        "systems_thinking": 0,
        "production_mindset": 0,
        "learning_velocity": 0,
        "research_maturity": 0
      }},
      "confidence_score": 0,
      "confidence_justification": "Explanation of confidence level",
      "reasoning": "Brief summary of candidate performance",
      "verdict": "Strong Hire/Hire/Maybe/No Hire"
    }}
    """
    
    try:
        response = await llm.chat(messages=[{"role": "system", "content": analysis_prompt}])
        report = json.loads(response.choices[0].message.content)
        
        # Weighted Math
        s = report["scores"]
        final_score = (
            (s["domain_thinking"] * 0.25) +
            (s["systems_thinking"] * 0.25) +
            (s["production_mindset"] * 0.20) +
            (s["learning_velocity"] * 0.20) +
            (s["research_maturity"] * 0.10)
        )
        report["final_weighted_score"] = round(final_score, 2)
        report["needs_manual_review"] = report["confidence_score"] < 5
        
        return report
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        return {"error": "Scoring failed", "details": str(e)}

# --- 4. SERVER & SESSION ---

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Using GPT-4o-mini for low-latency interview and high-quality analysis
    llm_instance = inference.LLM(model="openai/gpt-4o-mini")

    session = AgentSession(
        stt=deepgram.STT(model="nova-2-general"),
        llm=llm_instance,
        tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    async def finish_and_evaluate():
        """Runs once the session is disconnected."""
        try:
            history = session.history.to_dict()
            if not history:
                return

            print("--- SESSION ENDED: CALCULATING SCORE & CONFIDENCE ---")
            report = await generate_hiring_report(history, llm_instance)
            
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = transcript_dir / f"hiring_report_{ctx.room.name}_{current_date}.json"

            output_data = {
                "candidate": DUMMY_RESUME["candidate_info"]["name"],
                "evaluation": report,
                "transcript": history,
                "metadata": {"room": ctx.room.name, "timestamp": current_date}
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"REPORT SAVED: {filename.name}")
            print(f"FINAL SCORE: {report.get('final_weighted_score')} | CONFIDENCE: {report.get('confidence_score')}/10")
            print(f"VERDICT: {report.get('verdict')}")
            
        except Exception as e:
            logger.error(f"Error in shutdown callback: {e}")

    ctx.add_shutdown_callback(finish_and_evaluate)

    await session.start(
        agent=BaseThesisAssistant(DUMMY_RESUME),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=lambda p: noise_cancellation.BVC())
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)