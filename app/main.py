
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import predict_chunk, chunk_tokens, tokenizer
from app.postprocessing import build_response, merge_entities


app = FastAPI()


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if Path(file.filename or "").suffix.lower() != ".txt":
        raise HTTPException(
            status_code=415,
            detail="Only .txt files are supported",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded",
        )

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is empty",
        )

    tokens = tokenizer.tokenize(text)

    if len(tokens) <= 512:
        probabilities = predict_chunk(text)

        words = build_response(text, probabilities)
        entities = merge_entities(words)

        return entities

    else:
        response = {}

        chunks = chunk_tokens(text, 512, 20)

        for chunk in chunks:
            probabilities = predict_chunk(chunk)

            words = build_response(chunk, probabilities)
            entities = merge_entities(words)

            response.update(entities)

        return response

