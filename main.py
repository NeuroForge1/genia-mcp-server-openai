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

# --- Pydantic Models for Simplified MCP-like structure ---
class SimpleTextContent(BaseModel):
    text: str

class SimpleMessage(BaseModel):
    role: str
    content: SimpleTextContent
    metadata: Optional[Dict[str, Any]] = None

# --- Configuration ---
load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI(
    title="GENIA Simplified MCP Server - OpenAI",
    description="Servidor simplificado (sin librerías MCP externas) para interactuar con OpenAI vía SSE.",
)

# --- SSE Event Generator ---
async def openai_event_generator(request_message: SimpleMessage):
    """Generador de eventos SSE para la respuesta de OpenAI."""
    print(f"Servidor OpenAI Simplificado recibió: {request_message.model_dump_json()}")

    prompt_text = request_message.content.text
    model_name = request_message.metadata.get("model", "gpt-3.5-turbo") if request_message.metadata else "gpt-3.5-turbo"

    try:
        # Llamar a la API de OpenAI
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            stream=False # Mantener simple por ahora
        )

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            response_text = response.choices[0].message.content
            print(f"Respuesta de OpenAI: {response_text}")
            
            # Crear mensaje de respuesta simplificado
            response_msg = SimpleMessage(
                role="assistant",
                content=SimpleTextContent(text=response_text)
            )
            
            # Enviar respuesta como evento SSE
            yield {"event": "message", "data": response_msg.model_dump_json()}
            print("Respuesta enviada por SSE.")

        else:
            print("Error: Respuesta inesperada de OpenAI API")
            error_msg = SimpleMessage(
                role="assistant",
                content=SimpleTextContent(text="Error: No se recibió contenido en la respuesta de OpenAI.")
            )
            yield {"event": "error", "data": error_msg.model_dump_json()}

    except Exception as e:
        print(f"Error al llamar a OpenAI API: {e}")
        error_msg = SimpleMessage(
            role="assistant",
            content=SimpleTextContent(text=f"Error interno al procesar la solicitud con OpenAI: {e}")
        )
        yield {"event": "error", "data": error_msg.model_dump_json()}
    finally:
        # Señal de fin (opcional, pero útil)
        yield {"event": "end", "data": "Stream ended"}
        print("Stream SSE finalizado.")

# --- FastAPI Endpoint --- 
@app.post("/mcp")
async def handle_sse_request_post(request_message: SimpleMessage):
    """Endpoint SSE que recibe una solicitud MCP (simplificada) vía POST y devuelve la respuesta de OpenAI."""
    return EventSourceResponse(openai_event_generator(request_message))

@app.get("/")
async def root():
    return {"message": "Servidor MCP Simplificado para OpenAI activo. Endpoint SSE (POST) en /mcp"}

# (Opcional) Ejecutar con uvicorn directamente para pruebas rápidas
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)

