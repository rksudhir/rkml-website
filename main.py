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

@app.post("/predictPotatoLeaf")
async def predictPotatoLeaf(file: UploadFile = File(...)):
	try:
		MODEL = tf.keras.models.load_model("models/potato_model_2.keras")
		CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
		image = read_file_as_image(await file.read())
		img_batch = np.expand_dims(image, 0)
		predictions = MODEL.predict(img_batch)
		predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
		confidence = np.max(predictions[0])
		return f'<br/>&nbsp;- Disease: {predicted_class}<br/>&nbsp;- Confidence: {float(confidence)*100}%'  #{float(confidence)}'
	except Exception as error:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		# fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		return f"exc_type: {str(error)}, Line#: {exc_tb.tb_lineno}"
	
@app.post("/predictTomatoLeaf")
async def predictTomatoLeaf(file: UploadFile = File(...)):
	try:
		MODEL = tf.keras.models.load_model("models/tomato_model_1.keras")
		CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites or Two-spotted_spider_mite',
 'Tomato_Target_Spot',
 'Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato_mosaic_virus',
 'Tomato_healthy']
		
		image = read_file_as_image(await file.read())
		img_batch = np.expand_dims(image, 0)
		predictions = MODEL.predict(img_batch)
		# return f'Prediction: {str(predictions[0])} Gives: {str(np.argmax(predictions[0]))}'
		predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
		confidence = np.max(predictions[0])
		return f'<br/>&nbsp;- Disease: {predicted_class}<br/>&nbsp;- Confidence: {float(confidence)*100}%'  #{float(confidence)}'
	except Exception as error:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		# fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		return f"exc_type: {str(error)}, Line#: {exc_tb.tb_lineno}"

def read_file_as_image(data) -> np.ndarray:
	image = np.array(Image.open(BytesIO(data)))
	return image

if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=8000)