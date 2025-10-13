# FastAPI 서버리스 함수 (Vercel Python Runtime가 app 변수를 자동 인식)
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Tarot Backend", version="0.1.0")

# CORS (필요시 도메인 제한 걸면 됨)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 배포 후 프론트 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EchoIn(BaseModel):
    message: str

@app.get("/")
def root():
    return {"ok": True, "name": "tarot-backend", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/sum")
def sum_numbers(a: float | None = None, b: float | None = None):
    if a is None or b is None:
        raise HTTPException(status_code=400, detail="query string으로 a, b 값을 넘겨줘")
    return {"a": a, "b": b, "sum": a + b}

@app.post("/echo")
def echo(body: EchoIn):
    return {"echo": body.message}

# ---- 로컬 개발용 엔트리포인트 (uvicorn) ----
# 로컬에서 `python api/index.py`로도 띄울 수 있게 옵션 제공
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
