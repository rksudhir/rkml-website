from fastapi import FastAPI, File, UploadFile, Form, Body
from pydantic import BaseModel, Field, HttpUrl
from starlette.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
import sys, os
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier 

app = FastAPI()

@app.get("/ping/{name}")
async def helloBuddy(name):
	return f"Hello {name}, How are you?!"

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
# Request model for SimpleCropReco input data
class InputData(BaseModel):
	N: float
	P: float
	K: float
	temperature: float
	humidity: float	
	ph: float	
	rainfall: float	

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
		return f"Error: {str(error)}, Line#: {exc_tb.tb_lineno}"
	
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
		return f"Error: {str(error)}, Line#: {exc_tb.tb_lineno}"


@app.post("/predictPaddy")
async def predictTomatoLeaf(file: UploadFile = File(...)):
	try:
		MODEL = tf.keras.models.load_model("models/paddy_model_842.keras")
		CLASS_NAMES = ['bacterial_leaf_blight',
 'bacterial_leaf_streak',
 'bacterial_panicle_blight',
 'blast',
 'brown_spot',
 'dead_heart',
 'downy_mildew',
 'hispa',
 'normal',
 'tungro']
		
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
		return f"Error: {str(error)}, Line#: {exc_tb.tb_lineno}"


def read_file_as_image(data) -> np.ndarray:
	image = np.array(Image.open(BytesIO(data)))
	return image

@app.post("/predictCropReco")
async def predictCropReco(input_data: InputData):
	try:
		with open('models/simplecropreco_2.pkl', 'rb') as f:
			MODEL = pickle.load(f)

		# MODEL = tf.keras.models.load_model("models/classifier.pkl")
		CLASS_NAMES = ['apple', 'banana','blackgram','chickpea','coconut','coffee','cotton', 'grapes','jute', 'kidneybeans', 'lentil', 'maize','mango','mothbeans','mungbean','muskmelon','orange','papaya','pigeonpeas', 'pomegranate', 'rice', 'watermelon']
		features = np.array([[input_data.N, input_data.P, input_data.K, input_data.temperature, input_data.humidity, input_data.ph, input_data.rainfall]])
		predictions = MODEL.predict(features)
		ronum = predictions[0]
		predicted_class = CLASS_NAMES[ronum]
		# confidence = np.max(predictions[0])
		return f'<br/>&nbsp;- Recommended Crop: {predicted_class}' #<br/>&nbsp;- Confidence: {float(confidence)*100}%'  #{float(confidence)}'
	except Exception as error:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		return f"Error: {str(error)}, Line#: {exc_tb.tb_lineno} and N:{input_data.N}"

if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=8000)