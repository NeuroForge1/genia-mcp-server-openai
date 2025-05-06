# /home/ubuntu/genia_mcp_server_openai/main.py

import os
import json
import asyncio
from fastapi import FastAPI, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
import base64
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for Simplified MCP-like structure ---
class SimpleTextContent(BaseModel):
    text: str

class SimpleMessage(BaseModel):
    role: str
    content: SimpleTextContent
    metadata: Optional[Dict[str, Any]] = None

# --- Configuration ---
load_dotenv() # Load .env file for local development

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set. OpenAI requests will fail.")
    # Allow startup, but requests will fail later
    openai_client = None 
else:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="GENIA Simplified MCP Server - OpenAI",
    description="Servidor simplificado para interactuar con OpenAI (Chat & Whisper) vía SSE.",
)

# --- SSE Event Generator ---
async def openai_event_generator(request_message: SimpleMessage):
    """Generador de eventos SSE para la respuesta de OpenAI (Chat o Whisper)."""
    logger.info(f"Servidor OpenAI Simplificado recibió: {request_message.model_dump_json(exclude={	'metadata	': {	'parameters	': {	'audio_content_base64	'}}})}") # Exclude base64 from log

    capability = request_message.metadata.get("capability", "generate_text") if request_message.metadata else "generate_text"
    response_role = "assistant" # Default role for successful responses

    if not openai_client:
        error_msg = SimpleMessage(
            role="error",
            content=SimpleTextContent(text="Error interno: Cliente OpenAI no inicializado (falta API Key).")
        )
        yield {"event": "error", "data": error_msg.model_dump_json()}
        yield {"event": "end", "data": "Stream ended due to error"}
        return

    try:
        if capability == "generate_text":
            prompt_text = request_message.content.text
            model_name = request_message.metadata.get("model", "gpt-3.5-turbo") if request_message.metadata else "gpt-3.5-turbo"
            
            logger.info(f"Ejecutando capacidad: generate_text con modelo {model_name}")
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                stream=False # Keep simple for now
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                response_text = response.choices[0].message.content
                logger.info(f"Respuesta de OpenAI (Chat): {response_text[:100]}...")
            else:
                raise ValueError("Respuesta inesperada de OpenAI Chat API (sin contenido)")

        elif capability == "transcribe_audio":
            logger.info("Procesando capacidad: transcribe_audio")
            audio_path = None # Initialize audio_path
            temp_audio_file_created = False # Flag to track if temp file was created
            try:
                audio_content_b64 = request_message.metadata.get("parameters", {}).get("audio_content_base64") if request_message.metadata else None
                model_name = request_message.metadata.get("model", "whisper-1")
                language = request_message.metadata.get("parameters", {}).get("language") if request_message.metadata else None # Optional: ISO 639-1 language code

                if not audio_content_b64:
                    # Fallback: Check for file_path (though likely won't work in Render)
                    audio_path = request_message.metadata.get("parameters", {}).get("file_path") if request_message.metadata else None
                    logger.warning("No se encontró audio_content_base64, intentando usar file_path (puede fallar en Render).")
                    if not audio_path or not os.path.exists(audio_path):
                        raise ValueError("No se proporcionó audio_content_base64 ni una ruta de archivo válida.")
                else:
                    logger.info("Recibido audio_content_base64. Decodificando...")
                    try:
                        audio_bytes = base64.b64decode(audio_content_b64)
                        logger.info(f"Audio decodificado (bytes: {len(audio_bytes)}). Creando archivo temporal...")
                    except Exception as decode_err:
                        logger.error(f"Error al decodificar base64: {decode_err}")
                        raise ValueError(f"Error al decodificar audio base64: {decode_err}")
                    
                    # Create a temporary file to pass to OpenAI API
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio_file:
                         temp_audio_file.write(audio_bytes)
                         audio_path = temp_audio_file.name
                         temp_audio_file_created = True
                    logger.info(f"Contenido de audio guardado temporalmente en: {audio_path}")

                logger.info(f"Llamando a OpenAI Whisper API (modelo: {model_name}, archivo: {audio_path}, idioma: {language or 'auto'})...")
                
                with open(audio_path, "rb") as audio_file:
                    transcription_response = await openai_client.audio.transcriptions.create(
                        model=model_name,
                        file=audio_file,
                        language=language # Pass language if provided
                    )
                response_text = transcription_response.text
                logger.info(f"Respuesta de OpenAI (Whisper): {response_text[:100]}...")
            
            finally:
                # Clean up the temporary audio file ONLY if it was created here from base64
                if temp_audio_file_created and audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                        logger.info(f"Archivo temporal de audio (desde base64) eliminado: {audio_path}")
                    except OSError as e:
                        logger.error(f"Error al eliminar archivo temporal de audio {audio_path}: {e}")
        else:
            raise ValueError(f"Capacidad desconocida: {capability}")

        # Crear mensaje de respuesta simplificado
        response_msg = SimpleMessage(
            role=response_role,
            content=SimpleTextContent(text=response_text)
        )
        
        # Enviar respuesta como evento SSE
        yield {"event": "message", "data": response_msg.model_dump_json()}
        logger.info("Respuesta enviada por SSE.")

    except Exception as e:
        logger.exception(f"Error al procesar capacidad 	'{capability}	': {e}") # Use exception for stack trace
        error_msg = SimpleMessage(
            role="error", # Use 'error' role for exceptions
            content=SimpleTextContent(text=f"Error interno al procesar la solicitud ({capability}): {str(e)}")
        )
        yield {"event": "error", "data": error_msg.model_dump_json()}
    finally:
        # Señal de fin (opcional, pero útil)
        yield {"event": "end", "data": "Stream ended"}
        logger.info("Stream SSE finalizado.")

# --- FastAPI Endpoint --- 
@app.post("/mcp")
async def handle_sse_request_post(request_message: SimpleMessage):
    """Endpoint SSE que recibe una solicitud MCP (simplificada) vía POST y devuelve la respuesta de OpenAI."""
    return EventSourceResponse(openai_event_generator(request_message))

@app.get("/")
async def root():
    return {"message": "Servidor MCP Simplificado para OpenAI (Chat & Whisper) activo. Endpoint SSE (POST) en /mcp"}

# --- Run Server (for Render) ---
if __name__ == "__main__":
    import uvicorn
    # Render provides the PORT environment variable
    port = int(os.getenv("PORT", 8001)) # Default to 8001 if PORT not set
    logger.info(f"Iniciando servidor MCP de OpenAI en http://0.0.0.0:{port}")
    # Listen on 0.0.0.0 as required by Render
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) # Disable reload for production

