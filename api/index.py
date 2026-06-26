from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Tuple
from openai import OpenAI
import os
import json
import logging
import sys
import time
import threading
import hashlib

app = FastAPI(title="Dooray GPT Chat Bot")

# ---------- Logging ----------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(levelname)s %(asctime)s %(name)s : %(message)s",
)

logger = logging.getLogger("dooray-gpt-chat")

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY is not set.")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------- Conversation Store ----------
# 단일 프로세스 메모리 저장 방식입니다.
# Vercel serverless / 다중 worker 환경에서는 Redis, DB로 교체 권장.
CHAT_STORE: Dict[str, Dict[str, Any]] = {}
STORE_LOCK = threading.Lock()

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "1200"))


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
    response_type: str = "inChannel",
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


# ---------- Extract Helpers ----------
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


def extract_session_key(data: Dict[str, Any], req: Request) -> str:
    """
    Dooray payload 구조가 환경마다 다를 수 있어서 여러 후보값을 사용합니다.

    우선순위:
    1. 명시적 sessionId/channelId/threadId
    2. Dooray sender/channel 관련 필드
    3. IP 기반 fallback
    """
    candidate = (
        data.get("sessionId")
        or data.get("session_id")
        or data.get("threadId")
        or data.get("thread_id")
        or data.get("channelId")
        or data.get("channel_id")
        or data.get("roomId")
        or data.get("room_id")
        or data.get("userId")
        or data.get("user_id")
        or data.get("senderId")
        or data.get("sender_id")
    )

    if not candidate:
        user = data.get("user") or {}
        channel = data.get("channel") or {}

        if isinstance(user, dict):
            candidate = candidate or user.get("id") or user.get("name")

        if isinstance(channel, dict):
            channel_id = channel.get("id") or channel.get("name")
            if channel_id:
                candidate = f"{channel_id}:{candidate or 'unknown'}"

    if not candidate:
        client_host = req.client.host if req.client else "unknown"
        raw = f"{client_host}:{req.headers.get('user-agent', '')}"
        candidate = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    return str(candidate)


# ---------- Conversation Memory ----------
def get_or_create_chat(session_key: str) -> Dict[str, Any]:
    with STORE_LOCK:
        if session_key not in CHAT_STORE:
            CHAT_STORE[session_key] = {
                "summary": "",
                "messages": [],
                "createdAt": time.time(),
                "updatedAt": time.time(),
            }

        return CHAT_STORE[session_key]


def save_chat(session_key: str, chat: Dict[str, Any]):
    with STORE_LOCK:
        chat["updatedAt"] = time.time()
        CHAT_STORE[session_key] = chat


def build_messages(chat: Dict[str, Any], user_input: str) -> List[Dict[str, str]]:
    system_prompt = """
너는 Dooray 메신저에서 동작하는 한국어 대화형 GPT 봇이다.

규칙:
- 사용자의 이전 대화 맥락과 요약 기억을 참고해서 자연스럽게 답한다.
- 답변은 반드시 한국어로 한다.
- 답변은 반드시 500자를 넘기지 않는다.
- 너무 장황하게 설명하지 말고 핵심만 말한다.
- 모르면 추측하지 말고 필요한 부분을 짧게 확인한다.
- 코드 요청이면 바로 쓸 수 있게 핵심 코드나 수정 방향을 준다.
- 사용자의 말투가 짧으면 답변도 간결하게 맞춘다.
""".strip()

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    summary = chat.get("summary") or ""
    if summary:
        messages.append(
            {
                "role": "system",
                "content": f"이전 대화 요약 기억:\n{summary}",
            }
        )

    history = chat.get("messages") or []
    messages.extend(history[-MAX_HISTORY_MESSAGES:])

    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    return messages


def trim_answer(answer: str, max_chars: int = 500) -> str:
    answer = (answer or "").strip()

    if len(answer) <= max_chars:
        return answer

    return answer[:max_chars - 3].rstrip() + "..."


def summarize_if_needed(chat: Dict[str, Any]):
    """
    토큰 절약용 요약.
    메시지가 일정 개수 이상 쌓이면 오래된 대화를 summary로 압축하고,
    최근 메시지만 남깁니다.
    """
    messages = chat.get("messages") or []

    if len(messages) <= MAX_HISTORY_MESSAGES * 2:
        return

    old_messages = messages[:-MAX_HISTORY_MESSAGES]
    recent_messages = messages[-MAX_HISTORY_MESSAGES:]

    old_text = "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in old_messages
    )

    previous_summary = chat.get("summary") or ""

    prompt = f"""
아래 대화를 이후 대화에 필요한 기억만 남기도록 한국어로 짧게 요약해줘.
중요한 사용자 선호, 진행 중인 작업, 결정사항, 코드 구조만 유지해.
최대 {MAX_SUMMARY_CHARS}자 이내.

기존 요약:
{previous_summary}

추가 대화:
{old_text}
""".strip()

    try:
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "너는 대화 기록을 토큰 절약용으로 압축 요약하는 도우미다.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=1000,
        )

        summary = res.choices[0].message.content or ""
        chat["summary"] = summary[:MAX_SUMMARY_CHARS]
        chat["messages"] = recent_messages

    except Exception as e:
        logger.warning("[SUMMARY] failed: %s", e)
        chat["messages"] = recent_messages


def generate_chat_answer(session_key: str, question: str) -> str:
    chat = get_or_create_chat(session_key)
    messages = build_messages(chat, question)

    res = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=1000,
    )

    answer = res.choices[0].message.content or ""
    answer = trim_answer(answer, 500)

    chat_messages = chat.get("messages") or []

    chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    chat["messages"] = chat_messages

    summarize_if_needed(chat)
    save_chat(session_key, chat)

    return answer


def reset_chat(session_key: str):
    with STORE_LOCK:
        if session_key in CHAT_STORE:
            del CHAT_STORE[session_key]


# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "dooray-gpt-chat",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


@app.post("/dooray/gpt")
async def dooray_gpt(req: Request):
    """
    Dooray 대화형 GPT 엔드포인트.

    - 질문을 받으면 바로 GPT 답변 반환
    - 세션별 이전 대화 기록 유지
    - 오래된 대화는 요약해서 토큰 절약
    - 답변은 500자 이하로 유도 및 최종 절단
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
    session_key = extract_session_key(data, req)

    if not question:
        return respond(
            make_message(
                text="질문 내용을 입력해 주세요.",
                response_type="ephemeral",
            ),
            tag="gpt-empty",
        )

    if question.strip() in ["/reset", "reset", "초기화", "대화초기화"]:
        reset_chat(session_key)
        return respond(
            make_message(
                text="대화 기억을 초기화했습니다.",
                response_type="ephemeral",
            ),
            tag="gpt-reset",
        )

    try:
        answer = generate_chat_answer(session_key, question)

        return respond(
            make_message(
                text=answer,
                response_type="inChannel",
                replace_original=False,
            ),
            tag="gpt-chat",
        )

    except Exception as e:
        logger.exception("[GPT_CHAT] failed: %s", e)

        return respond(
            make_message(
                text="⚠️ GPT 응답 생성 중 오류가 발생했습니다. 로그를 확인해 주세요.",
                response_type="ephemeral",
            ),
            tag="gpt-chat-error",
        )


@app.post("/dooray/gpt/reset")
async def dooray_gpt_reset(req: Request):
    """
    현재 세션의 대화 기억 초기화.
    """
    verify_request(req)

    data, _ = await parse_dooray_payload(req)
    session_key = extract_session_key(data, req)

    reset_chat(session_key)

    return respond(
        make_message(
            text="대화 기억을 초기화했습니다.",
            response_type="ephemeral",
        ),
        tag="gpt-reset",
    )


@app.get("/dooray/gpt/debug")
async def dooray_gpt_debug(req: Request):
    """
    간단 디버그용.
    운영에서 필요 없으면 삭제해도 됩니다.
    """
    verify_request(req)

    with STORE_LOCK:
        return {
            "sessionCount": len(CHAT_STORE),
            "maxHistoryMessages": MAX_HISTORY_MESSAGES,
            "maxSummaryChars": MAX_SUMMARY_CHARS,
            "model": OPENAI_MODEL,
        }


# ----- Local run -----
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.index:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
