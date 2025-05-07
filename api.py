from fastapi import FastAPI, HTTPException
from api_functions import create_conversation, generate_audio, delete_conversation, generate_wav_file
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ConvRequest(BaseModel):
    text: str

# create conversation given context
# return: conversationID

@app.get("/create")
def createConversation():
    convId = create_conversation()
    return {"convesation_id": convId}


# generate audio - gen conversationID 
# return: audio

@app.post("/returnAudio/{audio_id}")
def generateConversation(audio_id: int, payload: ConvRequest):
    try:
        tensor = generate_audio(audio_id, payload.text)
        wav_io = generate_wav_file(tensor)
        return StreamingResponse(
            wav_io,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename=\"{audio_id}.wav\"'}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/delete/{audio_id}")
def deleteConversation(audio_id: int):
    try:
        status = delete_conversation(audio_id)
        return {f'Context {audio_id} was deleted'}
    except ValueError as e:
        raise HTTPException(status_code = 404, detail = str(e))
