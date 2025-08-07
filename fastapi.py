# main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import base64
from datetime import datetime
import asyncio

from conversation import ConversationAgent
from shopping_team import ShoppingTeam
from image_processing import ProductImageProcessingAgent

app = FastAPI(title="E-commerce Shopping Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global store for agent sessions
agents_store: Dict[str, Dict[str, Any]] = {}

# Pydantic Models
class ConfigRequest(BaseModel):
    api_key_llm: str
    api_key_search_tool: str
    api_key_firecrawl: str
    web_search_mode: str = "Tavily"
    llm_mode: str = "OpenAI"

class ChatRequest(BaseModel):
    session_id: str
    message: str
    image_data: Optional[str] = None

class ChatResponse(BaseModel):
    type: str  # "conversation" or "product_search"
    message: str
    products_html: Optional[str] = None
    continue_conversation: bool
    timestamp: str

class MessageModel(BaseModel):
    role: str
    content: str
    timestamp: str
    type: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    success: bool
    message: str

# Endpoints
@app.get("/")
async def root():
    return {"message": "E-commerce Shopping Assistant API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/config", response_model=SessionResponse)
async def setup_agents(config: ConfigRequest):
    # Validate required keys
    if not all([config.api_key_llm, config.api_key_search_tool, config.api_key_firecrawl]):
        raise HTTPException(status_code=400, detail="Missing required API keys")
    
    session_id = str(uuid.uuid4())
    
    try:
        # Initialize agents for this session
        agents_store[session_id] = {
            'conversation_agent': ConversationAgent(
                api_key=config.api_key_llm, 
                llm_mode=config.llm_mode
            ),
            'shopping_team': ShoppingTeam(
                api_key_llm=config.api_key_llm,
                api_key_search_tool=config.api_key_search_tool,
                search_tool=config.web_search_mode,
                llm_mode=config.llm_mode,
                firecrawl_api_key=config.api_key_firecrawl
            ),
            'image_processor': ProductImageProcessingAgent(
                api_key=config.api_key_llm, 
                llm_mode=config.llm_mode
            ),
            'messages': [],
            'created_at': datetime.now()
        }
        
        return SessionResponse(
            session_id=session_id,
            success=True,
            message="Agents initialized successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize agents: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def process_chat(chat_request: ChatRequest):
    session_id = chat_request.session_id
    
    if session_id not in agents_store:
        raise HTTPException(status_code=404, detail="Session not found. Please configure agents first.")
    
    agents = agents_store[session_id]
    user_input = chat_request.message
    
    try:
        # Step 1: Process image if provided
        if chat_request.image_data:
            # Decode base64 image
            image_bytes = base64.b64decode(chat_request.image_data.split(',')[1])
            
            # Process with image agent
            image_response = await asyncio.to_thread(
                agents['image_processor'].process_image,
                image_data=image_bytes,
                user_input=user_input
            )
            user_input = image_response.content
        
        # Add user message to history
        user_message = MessageModel(
            role="user",
            content=user_input,
            timestamp=datetime.now().isoformat()
        )
        agents['messages'].append(user_message.dict())
        
        # Step 2: Process with conversation agent
        conversation_response = await asyncio.to_thread(
            agents['conversation_agent'].process_query,
            user_input
        )
        
        # Step 3: Handle response based on conversation state
        if conversation_response['have_further_conversation']:
            # Continue conversation
            assistant_message = MessageModel(
                role="assistant",
                content=conversation_response["message"],
                timestamp=datetime.now().isoformat()
            )
            agents['messages'].append(assistant_message.dict())
            
            return ChatResponse(
                type="conversation",
                message=conversation_response["message"],
                continue_conversation=True,
                timestamp=datetime.now().isoformat()
            )
        
        else:
            # Add acknowledgment message
            assistant_message = MessageModel(
                role="assistant",
                content=conversation_response["message"],
                timestamp=datetime.now().isoformat()
            )
            agents['messages'].append(assistant_message.dict())
            
            # Trigger product search
            shopping_result = await asyncio.to_thread(
                agents['shopping_team'].run,
                payload=conversation_response["data"]
            )
            
            # Add product results to history
            product_message = MessageModel(
                role="assistant",
                content=shopping_result.content,
                type="product_results",
                timestamp=datetime.now().isoformat()
            )
            agents['messages'].append(product_message.dict())
            
            return ChatResponse(
                type="product_search",
                message=conversation_response["message"],
                products_html=shopping_result.content,
                continue_conversation=False,
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str):
    if session_id in agents_store:
        return {"messages": agents_store[session_id]['messages']}
    return {"messages": []}

@app.post("/api/clear/{session_id}")
async def clear_conversation(session_id: str):
    if session_id not in agents_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agents_store[session_id]['messages'] = []
        agents_store[session_id]['conversation_agent'].reset()
        return {"success": True, "message": "Conversation cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in agents_store:
        del agents_store[session_id]
        return {"success": True, "message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for session_id, data in agents_store.items():
        sessions.append({
            "session_id": session_id,
            "created_at": data["created_at"].isoformat(),
            "message_count": len(data["messages"])
        })
    return {"sessions": sessions}

# Optional: File upload endpoint for images
@app.post("/api/upload-image/{session_id}")
async def upload_image(session_id: str, file: UploadFile = File(...)):
    if session_id not in agents_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process with image agent
        agents = agents_store[session_id]
        image_response = await asyncio.to_thread(
            agents['image_processor'].process_image,
            image_data=file_content,
            user_input=""
        )
        
        return {
            "success": True,
            "extracted_text": image_response.content,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)