from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Translation request model
class TranslationRequest(BaseModel):
    text: str
    source_lang: str  # 'en' or 'ru'

# Load MarianMT models
MODEL_EN_RU = "Helsinki-NLP/opus-mt-en-ru"
MODEL_RU_EN = "Helsinki-NLP/opus-mt-ru-en"

tokenizer_en_ru = MarianTokenizer.from_pretrained(MODEL_EN_RU)
model_en_ru = MarianMTModel.from_pretrained(MODEL_EN_RU)

tokenizer_ru_en = MarianTokenizer.from_pretrained(MODEL_RU_EN)
model_ru_en = MarianMTModel.from_pretrained(MODEL_RU_EN)

# Translation function
def translate_text(text: str, source_lang: str) -> str:
    try:
        if source_lang == "en":
            inputs = tokenizer_en_ru(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model_en_ru.generate(**inputs)
            return tokenizer_en_ru.decode(outputs[0], skip_special_tokens=True)
        elif source_lang == "ru":
            inputs = tokenizer_ru_en(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model_ru_en.generate(**inputs)
            return tokenizer_ru_en.decode(outputs[0], skip_special_tokens=True)
        else:
            raise ValueError("Unsupported language")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Translation endpoint
@app.post("/translate")
def translate(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        translated_text = translate_text(request.text, request.source_lang)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the backend
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)