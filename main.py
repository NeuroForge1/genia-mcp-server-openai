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
import subprocess # Import subprocess for ffmpeg

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

# --- Helper Function for Conversion ---
def convert_ogg_to_wav(input_path: str, output_path: str) -> bool:
    """Converts an OGG file to WAV using ffmpeg."""
    try:
        command = [
            "ffmpeg",
            "-i", input_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le", # Standard WAV codec
            "-ar", "16000", # Standard sample rate for Whisper
            "-ac", "1", # Mono channel
            output_path
        ]
        # *** CORRECTED F-STRING SYNTAX ***
        logger.info(f"Ejecutando comando ffmpeg: {' '.join(command)}") 
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Conversión a WAV exitosa: {output_path}")
        logger.debug(f"ffmpeg stdout: {result.stdout}")
        logger.debug(f"ffmpeg stderr: {result.stderr}")
        return True
    except FileNotFoundError:
        logger.error("Error: ffmpeg no encontrado. Asegúrate de que esté instalado y en el PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error durante la conversión con ffmpeg: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado durante la conversión a WAV: {e}")
        return False

# --- SSE Event Generator ---
async def openai_event_generator(request_message: SimpleMessage):
    """Generador de eventos SSE para la respuesta de OpenAI (Chat o Whisper)."""
    # Log received message excluding potentially large base64 content
    log_safe_metadata = request_message.metadata.copy() if request_message.metadata else {}
    if log_safe_metadata.get("parameters", {}).get("audio_content_base64"):
        log_safe_metadata["parameters"]["audio_content_base64"] = f"<base64_content_omitted_length={len(log_safe_metadata["parameters"]["audio_content_base64"])}>"
    logger.info(f"Servidor OpenAI Simplificado recibió: role={request_message.role}, content=	'{request_message.content.text[:50]}...'	, metadata={log_safe_metadata}")

    # --- CORRECTED CAPABILITY SELECTION --- 
    capability = "generate_text" # Default capability
    if request_message.metadata:
        if "capability_name" in request_message.metadata:
            capability = request_message.metadata["capability_name"]
            logger.info(f"Capability identified from metadata.capability_name: {capability}")
        elif "capability" in request_message.metadata:
            capability = request_message.metadata["capability"]
            logger.info(f"Capability identified from metadata.capability: {capability}")
        else:
            logger.info("No capability specified in metadata, defaulting to generate_text")
    else:
        logger.info("No metadata provided, defaulting to generate_text")
    # --- END CORRECTION ---

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
            ogg_path = None
            wav_path = None
            temp_ogg_created = False
            temp_wav_created = False
            file_to_transcribe_path = None
            file_to_transcribe_name = None
            try:
                # Get parameters safely
                params = request_message.metadata.get("parameters", {}) if request_message.metadata else {}
                audio_content_b64 = params.get("audio_content_base64")
                model_name = request_message.metadata.get("model", "whisper-1") if request_message.metadata else "whisper-1"
                language = params.get("language") # Optional: ISO 639-1 language code

                if not audio_content_b64:
                    raise ValueError("No se proporcionó audio_content_base64.")
                
                # *** ADDED LOGGING FOR BASE64 LENGTH ***
                logger.info(f"Recibido audio_content_base64 (longitud: {len(audio_content_b64)}). Decodificando...")
                try:
                    audio_bytes = base64.b64decode(audio_content_b64)
                    # *** ADDED LOGGING FOR DECODED BYTES SIZE ***
                    logger.info(f"Audio decodificado (bytes: {len(audio_bytes)}). Creando archivo OGG temporal...")
                except Exception as decode_err:
                    logger.error(f"Error al decodificar base64: {decode_err}")
                    raise ValueError(f"Error al decodificar audio base64: {decode_err}")
                
                # Save original OGG temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_ogg_file:
                     temp_ogg_file.write(audio_bytes)
                     ogg_path = temp_ogg_file.name
                     temp_ogg_created = True
                logger.info(f"Contenido de audio OGG guardado temporalmente en: {ogg_path}")

                # Convert OGG to WAV
                wav_path = ogg_path.replace(".ogg", ".wav")
                logger.info(f"Intentando convertir {ogg_path} a {wav_path}...")
                conversion_success = convert_ogg_to_wav(ogg_path, wav_path)

                if conversion_success:
                    logger.info(f"Usando archivo WAV convertido para transcripción: {wav_path}")
                    file_to_transcribe_path = wav_path
                    file_to_transcribe_name = os.path.basename(wav_path)
                    temp_wav_created = True # Mark WAV as created for cleanup
                else:
                    logger.warning(f"Falló la conversión a WAV. Intentando transcribir el archivo OGG original: {ogg_path}")
                    file_to_transcribe_path = ogg_path
                    file_to_transcribe_name = os.path.basename(ogg_path)

                if not file_to_transcribe_path:
                     raise ValueError("No se pudo determinar la ruta del archivo de audio para la transcripción.")

                logger.info(f"Llamando a OpenAI Whisper API (modelo: {model_name}, archivo: {file_to_transcribe_path}, idioma: {language or 'auto'})...")
                
                # Use the file path (WAV if converted, OGG otherwise)
                with open(file_to_transcribe_path, "rb") as audio_file:
                    # Pass file as a tuple: (filename, file_object)
                    file_tuple = (file_to_transcribe_name, audio_file)
                    logger.info(f"Enviando archivo a Whisper API como tupla: {file_tuple[0]}")
                    transcription_response = await openai_client.audio.transcriptions.create(
                        model=model_name,
                        file=file_tuple, # Pass the tuple here
                        language=language # Pass language if provided
                    )
                response_text = transcription_response.text
                logger.info(f"Respuesta de OpenAI (Whisper): {response_text[:100]}...")
            
            finally:
                # Clean up temporary files
                if temp_wav_created and wav_path and os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                        logger.info(f"Archivo temporal WAV eliminado: {wav_path}")
                    except OSError as e:
                        logger.error(f"Error al eliminar archivo temporal WAV {wav_path}: {e}")
                if temp_ogg_created and ogg_path and os.path.exists(ogg_path):
                    try:
                        os.remove(ogg_path)
                        logger.info(f"Archivo temporal OGG eliminado: {ogg_path}")
                    except OSError as e:
                        logger.error(f"Error al eliminar archivo temporal OGG {ogg_path}: {e}")
        else:
            # Handle unknown capability explicitly
            logger.warning(f"Capacidad desconocida solicitada: {capability}")
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
        logger.exception(f"Error al procesar capacidad '{capability}': {e}") # Use exception for stack trace
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

