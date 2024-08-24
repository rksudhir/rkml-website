# rkml-website
For my ML projects
To run:
>fastapi run main.py

============================================
{StatusCode: 200, ReasonPhrase: 'OK', Version: 1.1, Content: System.Net.Http.HttpConnectionResponseContent, Headers:
{
  Date: Sun, 14 Jul 2024 15:44:40 GMT
  Server: uvicorn
  Content-Length: 78
  Content-Type: application/json
}}
------------------------------------------------
Id = 318, Status = RanToCompletion, Method = <<SendAsync>g__Core|83_0>d, Result = StatusCode: 200, ReasonPhrase: 'OK', Version: 1.1, Content: System.Net.Http.HttpConnectionResponseContent, Headers:
{
  Date: Sun, 14 Jul 2024 15:44:40 GMT
  Server: uvicorn
  Content-Length: 78
  Content-Type: application/json
}
Id = 320, Status = RanToCompletion, Method = <<SendAsync>g__Core|83_0>d, Result = StatusCode: 200, ReasonPhrase: 'OK', Version: 1.1, Content: System.Net.Http.HttpConnectionResponseContent, Headers:
{
  Date: Sun, 14 Jul 2024 16:50:30 GMT
  Server: uvicorn
  Content-Length: 79
  Content-Type: application/json
}

"Error: File format not supported: filepath=models/classifier.pkl. 
Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). 
Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. 
In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(models/classifier.pkl, call_endpoint='serving_default')` 
(note that your `call_endpoint` might have a different name)., Line#: 98 