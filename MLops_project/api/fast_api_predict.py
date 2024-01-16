from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
import os
import uvicorn
from http import HTTPStatus
import shutil
from MLops_project import predict_main

app = FastAPI()
condition = os.path.exists("/gcs")

models_directory = "gcs/dtu_mlops_grp33_processed_data/outputs/" if condition else "outputs/"
temp_dir = "temp_data/"
selected_model = None  # Global variable to store the selected model
model_name_to_path = {}


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
    global model_name_to_path
    model_name_to_path.clear()

    print(os.getcwd())
    print(models_directory)
    if "gcs" in models_directory:
        print(os.listdir("gcs/dtu_mlops_grp33_processed_data/"))
        print(os.listdir(models_directory))

    models = []
    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith(".ckpt"):
                models.append(os.path.join(root, file))

    html_content = "<h1>Select a Model</h1>"
    for model in models:
        path_parts = model.split(os.sep)
        # Assuming models_directory is the base, and the first two subfolders are date and time
        if len(path_parts) > 2:
            date, time = path_parts[-4], path_parts[-3]
            model_name = f"{date}_{time}_{os.path.basename(model)}"
        else:
            model_name = os.path.basename(model)

        model_name_to_path[model_name] = model
        html_content += f"<form action='/ui/set_model/' method='post'><button name='model_name' type='submit' value='{model_name}'>{model_name}</button></form>"
    return html_content


@app.get("/models/")
async def list_models_cli():
    response = {
        "current_directory": os.getcwd(),
        "models_directory": models_directory,
        "models": [],
    }

    if "gcs" in models_directory:
        response["gcs_contents"] = {
            "dtu_mlops_grp33_processed_data": os.listdir("gcs/dtu_mlops_grp33_processed_data/"),
            "models_directory_contents": os.listdir(models_directory),
        }

    for root, dirs, files in os.walk(models_directory):
        for file in files:
            if file.endswith(".ckpt"):
                full_path = os.path.join(root, file)
                path_parts = full_path.split(os.sep)
                if len(path_parts) > 2:
                    date, time = path_parts[-4], path_parts[-3]
                    model_name = f"{date}_{time}_{os.path.basename(file)}"
                else:
                    model_name = os.path.basename(file)

                response["models"].append({"name": model_name, "path": full_path})

    return response


@app.post("/ui/set_model/")
async def set_model(model_name: str = Form(...)):
    global selected_model
    # Use the model_name to get the actual model path from the mapping
    actual_model_path = model_name_to_path.get(model_name)
    if actual_model_path:
        selected_model = actual_model_path
        return {"message": f"Model set to {actual_model_path}"}
    else:
        return {"message": "Model not found"}


@app.get("/ui/predict/")
async def get_predict_page():
    global selected_model, model_name_to_path

    if selected_model is None:
        return HTMLResponse("<html><body><p>No model selected</p></body></html>", media_type="text/html")

    if not os.path.exists(selected_model):
        return HTMLResponse("<html><body><p>Model not found</p></body></html>", media_type="text/html")

    # Call your existing prediction function
    predictions = predict_main(selected_model, temp_dir)

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

    if not os.path.exists(selected_model):
        return {"error": "Model not found"}
    predictions = predict_main(selected_model, temp_dir)
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
