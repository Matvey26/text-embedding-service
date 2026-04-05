import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from model import init
from endpoints import router as endpoints_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель в фоне, сервер стартует сразу."""
    app.state.model_ready = False
    app.state.tokenizer = None
    app.state.model = None

    async def load_model():
        tokenizer, model = await asyncio.to_thread(init)
        app.state.tokenizer = tokenizer
        app.state.model = model
        app.state.model_ready = True

    asyncio.create_task(load_model())
    yield

    app.state.tokenizer = None
    app.state.model = None
    app.state.model_ready = False


app = FastAPI(
    title="rubert-mini-frida-embeddings",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(endpoints_router)
