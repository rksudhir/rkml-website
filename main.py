from fastapi import FastAPI, File, UploadFile, Form, Body
from pydantic import BaseModel, Field, HttpUrl
from starlette.responses import HTMLResponse
import uvicorn

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
		return f"Drinking my {chaytype} tea today"
	else:
		return "Bogus tea today :("
#---------------------------------------------------------------
@app.post("/files/")
async def create_file(file:bytes = File(...)):
	return {"filelength": len(file)}

@app.post("/uploadfile/")
async def create_uploadfile(file: UploadFile): 
	return {"filename": file.filename}

@app.post("/predict/")
async def predict(file: bytes = File(...)):
	return {'file length': len(file)}

if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=8000)