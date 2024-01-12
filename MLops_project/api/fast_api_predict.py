from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
import os
import uvicorn
from http import HTTPStatus
import shutil
from MLops_project import predict_main

app = FastAPI()
models_directory = "outputs/2024-01-10/13-00-17/models/"
temp_dir = "temp_data/"
selected_model = None  # Global variable to store the selected model


@app.get("/")
async def root():
    return FileResponse(os.path.join(os.getcwd(), "MLops_project/api/index.html"))


@app.get("/ui/")
async def ui_root():
    return HTMLResponse(
        """
    <html>
        <head><title>MLops Project UI</title></head>
        <body>
            <h1>MLops Project UI</h1>
            <ul>
                <li><a href="/ui/models/">List Models</a></li>
                <li><a href="/ui/images/">Upload Image</a></li>
                <li><a href="/ui/predict/">Predict</a></li>
            </ul>
        </body>
    </html>
    """
    )


@app.get("/ui/models/", response_class=HTMLResponse)
async def list_models():
    models = [f for f in os.listdir(models_directory) if f.endswith(".ckpt")]
    html_content = "<h1>Select a Model</h1>"
    for model in models:
        html_content += f"<form action='/ui/set_model/' method='post'><button name='model_name' type='submit' value='{model}'>{model}</button></form>"
    return html_content


@app.post("/ui/set_model/")
async def set_model(model_name: str = Form(...)):
    global selected_model
    selected_model = model_name
    return {"message": f"Model set to {model_name}"}


@app.get("/ui/predict/")
async def get_predict_page():
    if selected_model is None:
        return HTMLResponse("<html><body><p>No model selected</p></body></html>", media_type="text/html")

    model_path = os.path.join(models_directory, selected_model)
    if not os.path.exists(model_path):
        return HTMLResponse("<html><body><p>Model not found</p></body></html>", media_type="text/html")

    # Call your existing prediction function
    predictions = predict_main(model_path, temp_dir)

    # Format the predictions into HTML
    predictions_html = "<ul>" + "".join(f"<li>{pred}</li>" for pred in predictions) + "</ul>"
    return HTMLResponse(
        f"<html><body><h1>Predictions for model {selected_model}:</h1>{predictions_html}</body></html>",
        media_type="text/html",
    )


@app.get("/predict/")
async def predict_images():
    if selected_model is None:
        return {"error": "No model selected"}
    model_path = os.path.join(models_directory, selected_model)
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    predictions = predict_main(model_path, temp_dir)
    return predictions


@app.get("/ui/images/", response_class=HTMLResponse)
async def get_upload_page():
    return """
    <html>
        <head><title>Upload an Image</title></head>
        <body>
            <h1>Upload Image</h1>
            <form action="/images/" enctype="multipart/form-data" method="post">
                <input name="data" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """


@app.post("/images/")
async def image_upload(data: UploadFile = File(...)):
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, data.filename)
    with open(file_path, "wb") as image:
        content = await data.read()
        image.write(content)
    response = {
        "filename": data.filename,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
