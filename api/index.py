from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Any, Dict, Tuple, Optional
from openai import OpenAI
import os
import json
import logging
import sys
import uuid
import time
import threading

app = FastAPI(title="Dooray GPT Bot")

# ---------- Logging ----------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(levelname)s %(asctime)s %(name)s : %(message)s",
)

logger = logging.getLogger("dooray-gpt")

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY is not set.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------- In-memory GPT Result Store ----------
# 단일 프로세스 기준 저장소입니다.
# 운영에서 여러 worker/process/serverless 환경이면 Redis/DB로 바꾸는 것을 권장합니다.
GPT_RESULTS: Dict[str, Dict[str, Any]] = {}
LATEST_REQUEST_ID: Optional[str] = None
STORE_LOCK = threading.Lock()


# ---------- Common Response ----------
def respond(payload: Dict[str, Any], tag: str = "") -> JSONResponse:
    try:
        logger.info(
            "[RESP%s] %s",
            f"/{tag}" if tag else "",
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception:
        pass

    return JSONResponse(
        content=payload,
        media_type="application/json; charset=utf-8",
    )


def make_message(
    text: str,
    attachments=None,
    response_type: str = "ephemeral",
    replace_original: bool = False,
    delete_original: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text": text,
        "responseType": response_type,
        "replaceOriginal": replace_original,
        "deleteOriginal": delete_original,
    }

    if attachments:
        payload["attachments"] = attachments

    return payload


# ---------- Verify ----------
def verify_request(req: Request):
    expected = os.getenv("DOORAY_VERIFY_TOKEN")

    if not expected:
        return

    got = req.headers.get("X-Dooray-Token") or req.headers.get("Authorization")

    if got != expected:
        raise HTTPException(status_code=401, detail="invalid token")


# ---------- Payload Parser ----------
async def parse_dooray_payload(req: Request) -> Tuple[Dict[str, Any], bool]:
    ctype = (req.headers.get("content-type") or "").lower()

    # JSON 우선 처리
    if "application/json" in ctype:
        try:
            data = await req.json()

            is_action = bool(
                data.get("actionValue")
                or (
                    data.get("actions")
                    and isinstance(data.get("actions"), list)
                    and data["actions"]
                    and data["actions"][0].get("value")
                )
            )

            if not data.get("actionValue") and data.get("actions"):
                data["actionValue"] = data["actions"][0].get("value")
                data["actionName"] = data["actions"][0].get("name") or data.get("actionName")

            logger.info(
                "[PARSE/JSON] is_action=%s parsed=%s",
                is_action,
                json.dumps(data, ensure_ascii=False)[:2000],
            )

            return data, is_action

        except Exception as e:
            logger.warning("[PARSE/JSON] failed: %s", e)

    # FORM 처리
    form = await req.form()
    data: Dict[str, Any] = {}

    if "payload" in form:
        try:
            data = json.loads(form["payload"])
        except Exception:
            data = {}
    else:
        for k, v in form.items():
            if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                try:
                    data[k] = json.loads(v)
                    continue
                except Exception:
                    pass

            data[k] = v

    if data.get("actions") and isinstance(data["actions"], list) and data["actions"]:
        if not data.get("actionValue"):
            v = data["actions"][0].get("value")
            if v:
                data["actionValue"] = v

        if not data.get("actionName"):
            n = data["actions"][0].get("name")
            if n:
                data["actionName"] = n

    is_action = bool(data.get("actionValue"))

    logger.info(
        "[PARSE/FORM] is_action=%s parsed=%s",
        is_action,
        json.dumps(data, ensure_ascii=False)[:2000],
    )

    return data, is_action


def extract_question(data: Dict[str, Any]) -> str:
    question = (
        data.get("text")
        or data.get("question")
        or data.get("query")
        or data.get("message")
        or data.get("actionValue")
        or ""
    )

    return str(question).strip()


def extract_request_id(data: Dict[str, Any], req: Request) -> Optional[str]:
    request_id = (
        data.get("requestId")
        or data.get("request_id")
        or data.get("id")
        or data.get("text")
        or req.query_params.get("requestId")
        or req.query_params.get("request_id")
        or req.query_params.get("id")
    )

    if request_id is None:
        return None

    request_id = str(request_id).strip()

    return request_id or None


def save_result(
    request_id: str,
    status: str,
    question: str,
    answer: str = "",
    error: str = "",
):
    global LATEST_REQUEST_ID

    with STORE_LOCK:
        GPT_RESULTS[request_id] = {
            "requestId": request_id,
            "status": status,
            "question": question,
            "answer": answer,
            "error": error,
            "createdAt": GPT_RESULTS.get(request_id, {}).get("createdAt") or time.time(),
            "updatedAt": time.time(),
        }

        LATEST_REQUEST_ID = request_id


def get_result(request_id: Optional[str]) -> Optional[Dict[str, Any]]:
    with STORE_LOCK:
        rid = request_id or LATEST_REQUEST_ID

        if not rid:
            return None

        return GPT_RESULTS.get(rid)


def generate_gpt_answer(request_id: str, question: str):
    logger.info("[GPT] start request_id=%s question=%s", request_id, question[:500])

    try:
        prompt = f"{question}\n\n반드시 한국어로 대답해줘."

        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        answer = res.choices[0].message.content or ""

        save_result(
            request_id=request_id,
            status="done",
            question=question,
            answer=answer,
        )

        logger.info("[GPT] done request_id=%s answer_len=%d", request_id, len(answer))

    except Exception as e:
        logger.exception("[GPT] failed request_id=%s error=%s", request_id, e)

        save_result(
            request_id=request_id,
            status="error",
            question=question,
            error=str(e),
        )


# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "dooray-gpt",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


@app.post("/dooray/gpt")
async def dooray_gpt(req: Request, background_tasks: BackgroundTasks):
    """
    GPT 질문 접수 엔드포인트.

    동작:
    1. Dooray에서 질문 수신
    2. requestId 생성
    3. GPT 응답 생성을 백그라운드로 실행
    4. 이 엔드포인트에서는 GPT 답변을 바로 리턴하지 않음
    5. /dooray/gpt/result 에서 나중에 조회
    """
    verify_request(req)

    raw = (await req.body()).decode("utf-8", "ignore")
    logger.info(
        "[IN] POST /dooray/gpt CT=%s RAW=%s",
        req.headers.get("content-type"),
        raw[:2000],
    )

    data, _ = await parse_dooray_payload(req)
    question = extract_question(data)

    if not question:
        return respond(
            make_message(
                text="질문 내용을 입력해 주세요.",
                response_type="ephemeral",
            ),
            tag="gpt-empty",
        )

    request_id = str(uuid.uuid4())

    save_result(
        request_id=request_id,
        status="processing",
        question=question,
    )

    background_tasks.add_task(generate_gpt_answer, request_id, question)

    return respond(
        make_message(
            text=(
                "질의를 접수했습니다.\n"
                f"요청 ID: `{request_id}`\n\n"
                "응답은 잠시 후 결과 조회 엔드포인트에서 가져오면 됩니다."
            ),
            response_type="ephemeral",
            replace_original=False,
        ),
        tag="gpt-accepted",
    )


@app.post("/dooray/gpt/result")
async def dooray_gpt_result_post(req: Request):
    """
    GPT 결과 조회 엔드포인트 - POST 방식.

    Dooray slash command나 webhook에서 호출하기 좋게 POST 지원.
    text/requestId/id 중 하나로 requestId를 받을 수 있음.
    requestId가 없으면 가장 최근 요청 결과를 반환.
    """
    verify_request(req)

    raw = (await req.body()).decode("utf-8", "ignore")
    logger.info(
        "[IN] POST /dooray/gpt/result CT=%s RAW=%s",
        req.headers.get("content-type"),
        raw[:2000],
    )

    data, _ = await parse_dooray_payload(req)
    request_id = extract_request_id(data, req)
    result = get_result(request_id)

    return build_result_response(result)


@app.get("/dooray/gpt/result")
async def dooray_gpt_result_get(req: Request):
    """
    GPT 결과 조회 엔드포인트 - GET 방식.

    예:
    /dooray/gpt/result?id=요청ID
    /dooray/gpt/result?requestId=요청ID

    id/requestId가 없으면 가장 최근 요청 결과를 반환.
    """
    verify_request(req)

    request_id = extract_request_id({}, req)
    result = get_result(request_id)

    return build_result_response(result)


def build_result_response(result: Optional[Dict[str, Any]]) -> JSONResponse:
    if not result:
        return respond(
            make_message(
                text="조회 가능한 GPT 응답이 없습니다.",
                response_type="ephemeral",
            ),
            tag="gpt-result-empty",
        )

    status = result.get("status")
    request_id = result.get("requestId")
    question = result.get("question") or ""

    if status == "processing":
        return respond(
            make_message(
                text=(
                    "아직 GPT 응답을 생성 중입니다.\n"
                    f"요청 ID: `{request_id}`\n"
                    f"질문: {question}"
                ),
                response_type="ephemeral",
            ),
            tag="gpt-result-processing",
        )

    if status == "error":
        return respond(
            make_message(
                text=(
                    "⚠️ GPT 질의 중 오류가 발생했습니다.\n"
                    f"요청 ID: `{request_id}`\n"
                    f"오류: {result.get('error') or 'unknown error'}"
                ),
                response_type="ephemeral",
            ),
            tag="gpt-result-error",
        )

    answer = result.get("answer") or ""

    return respond(
        make_message(
            text=answer,
            response_type="inChannel",
            replace_original=False,
        ),
        tag="gpt-result-done",
    )


# ----- Local run -----
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.index:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
