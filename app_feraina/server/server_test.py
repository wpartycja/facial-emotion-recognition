from PIL import Image
import requests
import io

model_url = "http://localhost:8000/prediction"
path = 'path/to/test/image.jpg'
image = Image.open(path).convert('RGB')
buf = io.BytesIO()
image.save(buf, format='png')
byte_im = buf.getvalue()
files = {'file': ("some_useful_name", byte_im, "image/jpeg")}

predicted_mode = requests.post(model_url, files=files).json()["mode"]
print(predicted_mode)