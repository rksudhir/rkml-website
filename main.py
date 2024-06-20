from fastapi import FastAPI, File, UploadFile, Form, Body
from pydantic import BaseModel, Field, HttpUrl
from starlette.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
import sys, os

app = FastAPI()

MODEL = tf.keras.models.load_model("models/potato_model_2.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping/{name}")
async def helloBuddy(name):
	return f"Hello {name}, How are you?"

@app.get("/hello/")
async def pinging(count: int = 0):
	return f"Hey, I got only ${count}/- today!"

@app.get("/getchay/")
async def pinging2(chaytype: str| None = None):
	if chaytype:
		return f"Drinking my {chaytype} tea today!!!"
	else:
		return "Bogus tea today :("
#---------------------------------------------------------------
@app.post("/files/")
async def create_file(file:bytes = File(...)):
	bytes = await file.read()
	return {"filelength": len(file)}

@app.post("/uploadfile/")
async def create_uploadfile(file: UploadFile): 
	return {"filename": file.filename}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	try:
		image = read_file_as_image(await file.read())
		img_batch = np.expand_dims(image, 0)
		predictions = MODEL.predict(img_batch)
		predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
		confidence = np.max(predictions[0])
		return f'Class: {predicted_class}, Confidence: {float(confidence)*100}%'  #{float(confidence)}'
	except Exception as error:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		# fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		return f"exc_type: {str(error)}, Line#: {exc_tb.tb_lineno}"

def read_file_as_image(data) -> np.ndarray:
	image = np.array(Image.open(BytesIO(data)))
	return image

if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=8000)