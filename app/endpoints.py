import asyncio
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Literal
from model import tokenize, predict


class InputData(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Текст, для которого необходимо получить эмбеддинг",
    )
    prefix: Optional[
        Literal[
            "search_query",
            "search_document",
            "paraphrase",
            "categorize",
            "categorize_sentiment",
            "categorize_topic",
            "categorize_entailment",
        ]
    ] = Field(
        "categorize",
        description="Префикс (промпт), который тюнит эмбеддинг под конкретную задачу",
    )


router = APIRouter()


@router.post("/embed")
async def get_embeddings(data: InputData, request: Request):
    if not request.app.state.model_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not ready",
                "message": "Модель ещё загружается, попробуйте позже.",
            },
        )

    prefixed_text = f"{data.prefix}: {data.text}"

    tokenized_inputs = await asyncio.to_thread(
        tokenize, prefixed_text, request.app.state.tokenizer, truncation=False
    )
    token_count = tokenized_inputs["input_ids"].shape[1]

    if token_count > 2048:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input too long",
                "message": "Модель rubert-mini-frida не поддерживает больше 2048 токенов.",
                "token_count": int(token_count),
                "max_supported": 2048,
            },
        )

    warning = None
    if token_count > 512:
        warning = (
            f"WARNING: Количество токенов ({token_count}) превышает "
            f"рекомендуемый контекст модели (512 токенов)."
        )

    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"][:, :512]
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"][:, :512]

    embeddings = await asyncio.to_thread(
        predict, tokenized_inputs, request.app.state.model
    )

    response = {
        "embedding": embeddings[0],
        "token_count": min(512, int(token_count)),
    }

    if warning:
        response["warning"] = warning

    return response


@router.get("/health")
async def health_check(request: Request):
    if not request.app.state.model_ready:
        return {
            "status": "loading",
            "model": "rubert-mini-frida",
            "detail": "Model is still loading",
        }

    try:
        test_text = "ok"
        tokenized = await asyncio.to_thread(
            tokenize, test_text, request.app.state.tokenizer
        )
        emb = await asyncio.to_thread(predict, tokenized, request.app.state.model)

        if emb and len(emb[0]) > 0:
            return {
                "status": "healthy",
                "model": "rubert-mini-frida",
                "embedding_dim": len(emb[0]),
                "detail": "Model is loaded and working",
            }
        else:
            return {"status": "unhealthy", "detail": "Embedding generation failed"}

    except Exception as e:
        return {"status": "unhealthy", "detail": f"Error during health check: {str(e)}"}
