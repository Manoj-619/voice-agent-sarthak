import logging
from datetime import datetime
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Any

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
from livekit.plugins import noise_cancellation, silero , deepgram, cartesia , google
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from bithuman_voice import TTS  # ✅ Standalone package import


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
    print(f" Fallback directory: {transcript_dir.absolute()}")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a professional interviewer for ConnectionOS conducting a 
5-7 minute behavioral interview focused on problem-solving and decision-making.

INTERVIEW STRUCTURE:
1.⁠ ⁠Introduction (30s): Introduce yourself warmly
2.⁠ ⁠Question 1 (90s): "Tell me about a difficult problem you solved recently..."
3.⁠ ⁠Question 2 (90s): "Describe a time you made a decision with incomplete information..."
4.⁠ ⁠Question 3 (90s): "Tell me about a disagreement with a teammate..."
5.⁠ ⁠Closing (30s): Thank them and explain next steps

Keep it conversational, ask natural follow-ups, maintain 5-7 minute duration.""",
        )

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Metrics storage
    all_metrics: List[Any] = []
    
    session = AgentSession(
        # stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # stt=deepgram.STT(model="nova-2"),

        # llm=inference.LLM(model="openai/gpt-4o-mini"),
        # tts=cartesia.TTS(
        #     model="cartesia/sonic", 
        #     voice="a0e99841-438c-4a64-b679-82d9f9c9ce1a"
        # ),
        
       stt=deepgram.STT(model="nova-2-general"),
    #    llm=google.LLM(model="gemini-3-flash"),
        
       llm=inference.LLM(model="openai/gpt-4o-mini"),
#        tts=TTS(
#     model="apple",
#     voice="com.apple.speech.synthesis.voice.siri_male.en-US"  # Professional male
# ),

       tts=cartesia.TTS(
        model="sonic-3",
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc" ), # Professional male voice
    
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)  # Your exact code
        all_metrics.append(ev.metrics)
        logger.info(f"Metrics collected: {type(ev.metrics).__name__}")

    session.on("metrics_collected", _on_metrics_collected)

    async def write_transcript():
        try:
            # Directory ALREADY EXISTS from startup
            current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = transcript_dir / f"transcript_{ctx.room.name}_{current_date}.json"
            
            # Verify write permissions right before saving
            if not os.access(transcript_dir, os.W_OK):
                print(f"NO WRITE PERMISSION: {transcript_dir}")
                print(f"Run: chmod 755 {transcript_dir}")
                return
            
            history_dict = session.history.to_dict()
            
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
                "transcript": history_dict,
                "latency_metrics": latency_data,
                "room_name": ctx.room.name,
                "timestamp": current_date
            }
            
            # FINAL WRITE with error catching
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write
            
            print(f"SAVED: {filename.absolute()}")
            print(f"Total latency: {total_latency:.3f}s ({len(all_metrics)} metrics)")
            
        except PermissionError:
            print(f"PERMISSION DENIED: {transcript_dir}")
            print(f"Fix: chmod -R 755 {transcript_dir.parent}")
            print(f"Or: sudo mkdir -p {TRANSCRIPT_PATH} && sudo chown $USER {TRANSCRIPT_PATH}")
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

if __name__ == "__main__":
    cli.run_app(server)
