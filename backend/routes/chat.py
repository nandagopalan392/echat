# ...existing code...

# Update imports to use lazy loading
from main import get_rag  # Replace direct import of RAG

# ...existing code...

@router.post("/send")
async def send_message(
    request: Request,
    message: ChatMessage,
    current_user: User = Depends(get_current_user)
):
    # ...existing code for session management...
    
    # Only load RAG when needed
    rag = get_rag()  # This will only initialize RAG on demand

    # Create a streaming response
    return StreamingResponse(
        stream_response(message.content, session_id, current_user),
        media_type="text/event-stream"
    )

async def stream_response(message_content, session_id, user):
    # ...existing code for preparing the message...
    
    try:
        # Get the chat history for context
        # ...existing code...
        
        # Prepare message for streaming
        full_response = ""
        is_thinking = False
        thinking_content = ""
        
        # Use stream=True to get token-by-token response from Ollama
        async for token in ollama_client.stream(
            model=llm_model_name,
            prompt=message_content,
            context=context_ids,
            options={"temperature": 0.7},
        ):
            if token.get("done", False):
                # Send the final complete message with proper SSE formatting
                data = json.dumps({'is_final': True, 'full_response': full_response, 'session_id': session_id})
                yield f"data: {data}\n\n"
                break
                
            # Get the individual token
            if "response" in token:
                token_text = token["response"]
                full_response += token_text
                
                # Track think blocks for special handling
                if "<think>" in token_text and not is_thinking:
                    is_thinking = True
                    thinking_content = ""
                elif "</think>" in token_text and is_thinking:
                    is_thinking = False
                
                if is_thinking:
                    thinking_content += token_text
                
                # Send each token to the frontend with proper SSE formatting
                data = json.dumps({'token': token_text, 'full_response': full_response, 'is_thinking': is_thinking})
                yield f"data: {data}\n\n"
                
                # Small delay to control streaming speed if needed
                await asyncio.sleep(0.01)
        
        # Save the message and response to the database
        # ...existing code...
        
    except Exception as e:
        logging.error(f"Error in stream_response: {str(e)}")
        data = json.dumps({'error': str(e)})
        yield f"data: {data}\n\n"

# ...existing code...
