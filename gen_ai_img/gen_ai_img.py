import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Initialize the Gemini Client
# Make sure GEMINI_API_KEY is in your .env file
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    prompt = request.form.get('prompt')
    file = request.files.get('image')

    try:
        contents = [prompt]

        # If a file is uploaded, it's an "Edit" task (Image-to-Image)
        if file:
            img = Image.open(file.stream)
            contents.append(img)

        # Call Nano Banana (Gemini 2.5 Flash Image)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            )
        )

        # Extract the image from the response
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                # Convert bytes to base64 for web display
                img_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                return jsonify({"image": f"data:image/png;base64,{img_base64}"})

        return jsonify({"error": "No image was generated"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)