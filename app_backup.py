from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file
from PIL import Image
import os
import uuid
import shutil

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def valid_image(file):
    if not file or not file.filename or "." not in file.filename:
        return False

    if file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED_EXTENSIONS:
        return False

    try:
        file.stream.seek(0)
        Image.open(file.stream).verify()
        file.stream.seek(0)
        return True
    except Exception:
        file.stream.seek(0)
        return False


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        person = request.files.get("person")
        cloth = request.files.get("cloth")

        if not valid_image(person) or not valid_image(cloth):
            return render_template(
                "index.html",
                error="Upload valid JPG, PNG or WEBP images."
            )

        person_name = uuid.uuid4().hex + "_" + secure_filename(person.filename)
        cloth_name = uuid.uuid4().hex + "_" + secure_filename(cloth.filename)

        person_path = os.path.join(UPLOAD_FOLDER, person_name)
        cloth_path = os.path.join(UPLOAD_FOLDER, cloth_name)

        person.save(person_path)
        cloth.save(cloth_path)

        try:
            client = Client("yisol/IDM-VTON")

            result = client.predict(
                dict={
                    "background": handle_file(person_path),
                    "layers": [],
                    "composite": None
                },
                garm_img=handle_file(cloth_path),
                garment_des="uploaded garment",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )

            image_result = result[0]
            if isinstance(image_result, dict):
                image_result = image_result.get("path")

            if not image_result or not os.path.isfile(image_result):
                raise RuntimeError(f"Unexpected AI result: {result}")

            output_name = "tryon_" + uuid.uuid4().hex + ".png"
            shutil.copy2(
                image_result,
                os.path.join(OUTPUT_FOLDER, output_name)
            )

            return render_template(
                "index.html",
                person_image=person_name,
                cloth_image=cloth_name,
                result_image=output_name,
                message="AI try-on completed!"
            )

        except Exception:
            app.logger.exception("AI try-on failed")
            return render_template(
                "index.html",
                person_image=person_name,
                cloth_image=cloth_name,
                error="AI try-on failed. Check the VS Code Terminal."
            )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/outfit-ideas", methods=["POST"])
def outfit_ideas():
    data = request.get_json() or {}

    occasion = data.get("occasion", "Casual")
    style = data.get("style", "Smart Casual")
    color = data.get("color", "Black")

    ideas = {
        "Casual": f"{color} T-shirt, blue jeans and white sneakers.",
        "College": f"{color} polo shirt, straight-fit jeans and clean sneakers.",
        "Party": f"{color} shirt, black trousers and stylish shoes.",
        "Wedding": f"{color} kurta, cream trousers and formal loafers.",
        "Office": f"{color} shirt, tailored trousers and formal shoes."
    }

    suggestion = ideas.get(occasion, ideas["Casual"])

    return jsonify({
        "occasion": occasion,
        "suggestion": suggestion,
        "style": style
    })


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)