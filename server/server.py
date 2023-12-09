import uvicorn
from typing import Annotated
from fastapi import FastAPI, UploadFile, File

from PIL import Image
from io import BytesIO

from request_body import ModeResponse, Testing

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)
from new_demo import Model


app = FastAPI()

if __name__ == "__main__":

    model = Model()
    todos = ['uno', 'dos', 'tres']

    @app.post("/prediction")
    async def prediction(file: UploadFile = File(...)):
        request_object_content = await file.read()
        print("ok")
        image = Image.open(BytesIO(request_object_content)).convert('RGB')
        mode = model.fer(image)
        return ModeResponse(mode=mode)
    

    @app.post("/wrr")
    async def pre(file: Annotated[bytes, File()]):
        # request_object_content = await file.read()
        print("ok")
        print(file[0])
        print(file[1])
        # image = Image.open(BytesIO(file[0])).convert('RGB')
        # mode = model.fer(image)
        # return ModeResponse(mode=mode)
        return ModeResponse(mode="hhhh")

    @app.post("/string")
    async def pre(writing: str):
        # request_object_content = await file.read()
        # print("ok")
        # print(file[0])
        # print(file[1])
        # image = Image.open(BytesIO(file[0])).convert('RGB')
        # mode = model.fer(image)
        # return ModeResponse(mode=mode)
        return ModeResponse(mode="Good job!")
    
    @app.post("/test")
    async def pre(content):
        # request_object_content = await file.read()
        # print("ok")
        # print(file[0])
        # print(file[1])
        # image = Image.open(BytesIO(file[0])).convert('RGB')
        # mode = model.fer(image)
        # return ModeResponse(mode=mode)
        return ModeResponse(mode="Good job!")
    
    @app.get("/todo", tags=["todos"])
    async def get_todos() -> dict:
        return { "data": todos }


    @app.post("/todo")
    async def add_todo(todo: dict) -> dict:
        todos = [todo]
        return {
            "data": { "Todo added." }
        }


    uvicorn.run(app, host='0.0.0.0', port=8000)
