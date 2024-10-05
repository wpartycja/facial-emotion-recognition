from pydantic import BaseModel

class EmotionResponse(BaseModel):
    emotion: str
