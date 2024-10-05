import uvicorn
import onnxruntime
import numpy as np

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

from request_body import EmotionResponse


def preprocess_image(bytes_io) -> np.array:
    """
    Opens and prepares image for the model.
    It mimics original torch transform.Compose.
    Steps:
        1. open the image
        2. resize image to 224x224
        3. change shape to (3, 224, 224) from (224, 224, 3)
        4. change values from 0-255 to 0-1
        5. normalize image
        6. add batch dimension
    """
    img = Image.open(bytes_io).convert('RGB')                        # 1
    resized_img = np.array(img.resize((224, 224), Image.BILINEAR))   # 2
    transposed_img = resized_img.transpose((2, 0, 1))                # 3
    normalized_img = transposed_img / 255.0                          # 4
    mean = np.array([0.485, 0.456, 0.406])                           # 5
    std = np.array([0.229, 0.224, 0.225])
    normalized_img = (normalized_img - mean[:, None, None]) / std[:, None, None]
    input_data = normalized_img.reshape((1, 3, 224, 224))            # 6
    input_data = input_data.astype(np.float32)

    return input_data


def label_dict(model_type):
    """
    Depending of model type, the labels for classes are different.
    """
    expw_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    affect_labels = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'angry'} 
    raf_labels = {0: 'surprise', 1: 'fear', 2: 'disgust', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'neutral'} 
    labels_dict = {'raf': raf_labels, 'affect': affect_labels, 'expw': expw_labels}
    return labels_dict[model_type]


app = FastAPI()


if __name__ == "__main__":

    onnx_model_path = './raf_best.onnx'
    onnx_session = onnxruntime.InferenceSession(onnx_model_path)

    model_type = 'raf'                  # raf / expw / affect
    labels = label_dict(model_type)

    @app.post("/prediction")
    async def prediction(file: UploadFile = File(...)):
        request_object_content = await file.read()
        input_image = preprocess_image(BytesIO(request_object_content))
        output = onnx_session.run(None, {'x.1': input_image})
        pred = np.argmax(output)
        pred_label = labels[pred]

        return EmotionResponse(emotion=pred_label)

    uvicorn.run(app, host='0.0.0.0', port=8000)
