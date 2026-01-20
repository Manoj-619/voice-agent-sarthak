import logging

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
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)

# import logging
# import json
# from datetime import datetime
# from pathlib import Path

# from dotenv import load_dotenv
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
# )
# from livekit.plugins import noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("agent")
# logging.basicConfig(level=logging.INFO)

# load_dotenv(".env.local")


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
#             You eagerly assist users with their questions by providing information from your extensive knowledge.
#             Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
#             You are curious, friendly, and have a sense of humor.""",
#         )


# class TranscriptLogger:
#     """Handles saving conversation transcripts to a file"""
    
#     def __init__(self, room_name: str, output_dir: str = "transcripts"):
#         self.room_name = room_name
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True)
        
#         # Create a unique filename with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.filename = self.output_dir / f"transcript_{room_name}_{timestamp}.txt"
        
#         self.messages = []
#         logger.info(f"Transcript will be saved to: {self.filename}")
    
#     def add_message(self, role: str, text: str):
#         """Add a message to the transcript"""
#         if not text or not text.strip():
#             return
            
#         timestamp = datetime.now().isoformat()
#         message = f"[{timestamp}] {role.upper()}: {text}"
#         self.messages.append(message)
#         logger.info(f"Transcript: {message}")
    
#     def save(self):
#         """Save transcript to file"""
#         try:
#             with open(self.filename, "w", encoding="utf-8") as f:
#                 if self.messages:
#                     f.write("\n".join(self.messages))
#                     logger.info(f"Transcript saved to {self.filename} ({len(self.messages)} entries)")
#                 else:
#                     f.write("No transcript entries captured.\n")
#                     logger.warning(f"Transcript file created but empty: {self.filename}")
#         except Exception as e:
#             logger.error(f"Error saving transcript to {self.filename}: {e}")


# server = AgentServer()


# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()


# server.setup_fnc = prewarm


# @server.rtc_session()
# async def my_agent(ctx: JobContext):
#     # Logging setup
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }
    
#     # Initialize transcript logger
#     transcript_logger = TranscriptLogger(ctx.room.name)

#     # Set up a voice AI pipeline
#     session = AgentSession(
#         stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
#         llm=inference.LLM(model="openai/gpt-4.1-mini"),
#         tts=inference.TTS(
#             model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
#         ),
#         turn_detection=MultilingualModel(),
#         vad=ctx.proc.userdata["vad"],
#         preemptive_generation=True,
#     )

#     # Start the session FIRST
#     await session.start(
#         agent=Assistant(),
#         room=ctx.room,
#         room_options=room_io.RoomOptions(
#             audio_input=room_io.AudioInputOptions(
#                 noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
#                 if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
#                 else noise_cancellation.BVC(),
#             ),
#         ),
#     )

#     # Register event handlers AFTER session.start()
#     # Try multiple event name variations to find what works
#     @session.on("user_speech")
#     def on_user_speech(event):
#         """Called when user speech is transcribed"""
#         text = getattr(event, 'text', None) or getattr(event, 'message', None) or str(event)
#         if text and text.strip():
#             logger.info(f"User said: {text}")
#             transcript_logger.add_message("user", text)
    
#     @session.on("agent_speech")
#     def on_agent_speech(event):
#         """Called when agent speech is generated"""
#         text = getattr(event, 'text', None) or getattr(event, 'message', None) or str(event)
#         if text and text.strip():
#             logger.info(f"Agent said: {text}")
#             transcript_logger.add_message("assistant", text)

#     # Also try listening to STT events directly
#     if hasattr(session, 'stt') and session.stt:
#         @session.stt.on("transcription")
#         def on_stt_transcription(event):
#             text = getattr(event, 'text', None) or getattr(event, 'transcript', None)
#             if text and text.strip():
#                 logger.info(f"STT transcription: {text}")
#                 transcript_logger.add_message("user", text)

#     # Join the room and connect to the user
#     await ctx.connect()
    
#     # Wait for the session to end
#     try:
#         # Wait for session to complete
#         await session.aclose()
#     except Exception as e:
#         logger.error(f"Session error: {e}")
#     finally:
#         # Save transcript when session ends
#         transcript_logger.save()
#         logger.info("Session ended, transcript saved")


# if __name__ == "__main__":
#     cli.run_app(server)


