from fastapi import FastAPI
from api_functions import create_conversation, generate_audio, generate_wav_file
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ConvRequest(BaseModel):
    audioId: int
    text: str

# create conversation given context
# return: conversationID

@app.get("/create")
def createConversation():
    convId = create_conversation()
    return convId


# generate audio - gen conversationID 
# return: audio

@app.post("/returnAudio")
def generateConversation(payload: ConvRequest):
    try:
        tensor = generate_audio(payload.audioId, payload.text)
        wav_io = generate_wav_file(tensor)
        return StreamingResponse(
            wav_io,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename=\"{payload.audioId}.wav\"'}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
        
