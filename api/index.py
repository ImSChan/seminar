from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple
from openai import OpenAI
import os, json, random
import logging, sys
from fastapi.responses import JSONResponse

app = FastAPI(title="Dooray Tarot Bot")

# ---------- 로깅 기본 설정 (stderr가 아닌 stdout로) ----------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(levelname)s %(asctime)s %(name)s : %(message)s",
)

logger = logging.getLogger("dooray-tarot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# -------- OpenAI Client --------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- Load Card Keywords --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # 프로젝트 루트
KEYWORD_PATH = os.path.join(BASE_DIR, "data", "card_keywords.json")
with open(KEYWORD_PATH, "r", encoding="utf-8") as f:
    CARD_KEYWORDS: Dict[str, str] = json.load(f)

# -------- Static URL Builder --------
def public_url(request: Request, path: str) -> str:
    """
    /public 하위 파일을 정적 URL로.
    path는 '/card/xxx.jpg' 처럼 전달.
    """
    base = os.getenv("APP_BASE_URL")
    if not base:
        scheme = request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        base = f"{scheme}://{host}"
    return f"{base}{path}"

# -------- Utility: attachments builders --------
def make_message(
    text: str,
    attachments: List[Dict[str, Any]] = None,
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

def attachment_text_block(text: str) -> Dict[str, Any]:
    return {"text": text}

def attachment_image_block(
    title: str, image_url: str, thumb_url: str = None,
    author_name: str = None, title_link: str = None,
    callback_id: str = None
) -> Dict[str, Any]:
    block: Dict[str, Any] = {"title": title, "imageUrl": image_url}
    if thumb_url: block["thumbUrl"] = thumb_url
    if author_name: block["authorName"] = author_name
    if title_link: block["titleLink"] = title_link
    if callback_id: block["callbackId"] = callback_id
    return block

def attachment_fields_block(fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"fields": fields}

def attachment_actions_block(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"actions": actions}

def action_button(text: str, name: str, value: str, style: str = "primary") -> Dict[str, Any]:
    return {"type": "button", "text": text, "name": name, "value": value, "style": style}

# -------- Dooray payload parser (JSON + FORM) --------
async def parse_dooray_payload(req: Request) -> Tuple[Dict[str, Any], bool]:
    """
    Dooray의 Slash/Action 페이로드를 JSON/FORM 모두 지원해서 파싱.
    return: (data, is_action)
    """
    ctype = (req.headers.get("content-type") or "").lower()

    # 1) JSON 시도
    if "application/json" in ctype:
        try:
            data = await req.json()
            # 액션 여부: 최상위 actionValue 또는 actions[0].value 로 판단
            is_action = bool(
                data.get("actionValue")
                or (data.get("actions") and data["actions"][0].get("value"))
            )
            # actions[0].value 형태면 actionValue로 승격시켜 일관 처리
            if not data.get("actionValue") and data.get("actions"):
                data["actionValue"] = data["actions"][0].get("value")
                data["actionName"]  = data["actions"][0].get("name") or data.get("actionName")
            return data, is_action
        except Exception:
            pass  # 폼 전송일 가능성

    # 2) FORM 시도 (x-www-form-urlencoded, multipart/form-data)
    form = await req.form()
    data: Dict[str, Any] = {}

    # Slack류처럼 payload에 JSON 문자열이 통째로 들어오는 케이스
    if "payload" in form:
        try:
            data = json.loads(form["payload"])
        except Exception:
            data = {}
    else:
        # 키-값 그대로 받기
        for k, v in form.items():
            # originalMessage 같은 중첩 JSON 필드일 수 있음 → 파싱 시도
            if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                try:
                    data[k] = json.loads(v)
                    continue
                except Exception:
                    pass
            data[k] = v

    # actions[0].value → actionValue로 승격
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
    return data, is_action

# -------- GPT Helpers --------
SPREAD_FILES = {
    1:  "card_1.png",
    2:  "card_2.png",  # 추가
    3:  "card_3.png",
    5:  "card_5.png",
    6:  "card_6.png",
    10: "card_10.png",
}

def decide_spread(topic: str) -> Dict[str, Any]:
    system_prompt = """
너는 숙련된 타로 마스터야. 사용자의 질문을 분석해서 다음 중 어떤 타로 스프레드(배열)를 사용할지 결정해줘.
- 1장: 간단한 조언
- 2장: 선택지 비교, 양자택일  
- 3장: 과거-현재-미래
- 5장: 갈등/결정 분석
- 6장: 관계나 연애 분석
- 10장: 인생, 진로 등 복잡한 주제

JSON 형식만 반환(코드블록 금지):
{"spread":"3장","reason":"...","card_count":3}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":system_prompt.strip()},
                {"role":"user","content":topic}
            ],
            temperature=0.7,
        )
        text = res.choices[0].message.content.strip()
        return json.loads(text)
    except Exception:
        # 실패 시 기본값(3장)
        return {"spread":"3장","reason":"기본값으로 과거-현재-미래 흐름 확인","card_count":3}

def gpt_card_reading(cards: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
    """
    cards = [{"name": "바보 카드.jpg", "reversed": True}, ...]
    반환: {"items":[{name,position,keyword,meaning,advice},...],"summary":"..."}
    """
    cards_input = []
    for c in cards:
        name = c["name"].replace(".jpg", "")
        keyword = CARD_KEYWORDS.get(c["name"], "키워드 없음")
        position = "역방향" if c["reversed"] else "정방향"
        cards_input.append({"name": name, "position": position, "keyword": keyword})

    prompt = f"""
당신은 숙련된 타로카드 리더입니다.
'{topic}' 주제로 사용자가 아래 카드를 뽑았습니다.
각 카드의 이름/방향/키워드가 주어집니다:

{json.dumps(cards_input, ensure_ascii=False, indent=2)}

다음 JSON 형식으로만 답변하세요(코드블록 금지):
{{
  "items": [
    {{
      "name": "카드 이름",
      "position": "정방향/역방향",
      "keyword": "간단 키워드",
      "meaning": "카드 해석",
      "advice": "사용자 조언"
    }}
  ],
  "summary": "🧙 전체 해석: ..."
}}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt.strip()}],
            temperature=0.8,
        )
        txt = res.choices[0].message.content.strip()
        return json.loads(txt)
    except Exception:
        # 파싱 실패 시 통째로 텍스트를 summary로
        return {"items": [], "summary": "🧙 전체 해석: 카드의 조합을 긍정적으로 바라보되, 핵심에 집중하세요."}

# -------- UI Builder --------
def build_pick_ui(
    request: Request, count: int, picked: List[int], seed: int, topic: str,
    response_type: str = "ephemeral", replace_original: bool = False
) -> dict:
    spread_file = SPREAD_FILES.get(count, SPREAD_FILES[3])
    spread_img_url = public_url(request, f"/card_spread/{spread_file}")

    picked_str = ", ".join(map(str, picked)) if picked else "없음"
    remain = [i for i in range(1, count + 1) if i not in picked]

    atts: List[Dict[str, Any]] = []
    atts.append({"callbackId": "tarot-pick",
                 "text": f"🃏 번호를 순서대로 **{count}개** 선택해줘요.\n현재 선택: **{picked_str}**",
                 "title": f"{count}장 스프레드",
                 "imageUrl": spread_img_url})

    # 번호 버튼 (5개씩 끊어서 rows 구성)
    row: List[Dict[str, Any]] = []
    for i, num in enumerate(remain, start=1):
        row.append({"type": "button", "text": str(num), "name": "pick",
                    "value": f"pick|{count}|{seed}|{','.join(map(str,picked))}|{num}",
                    "style": "default"})
        if i % 5 == 0:
            atts.append({"callbackId": "tarot-pick", "actions": row})
            row = []
    if row:
        atts.append({"callbackId": "tarot-pick", "actions": row})

    # 리셋/랜덤 채우기
    atts.append({
        "callbackId": "tarot-pick",
        "actions": [
            {"type": "button", "text": "🔄 다시 선택", "name": "reset",
             "value": f"reset|{count}|{seed}|{','.join(map(str,picked))}|{topic}", "style": "default"},
            {"type": "button", "text": "🎲 무작위로 채우기", "name": "fill",
             "value": f"fill|{count}|{seed}|{','.join(map(str,picked))}|{topic}", "style": "default"}
        ]
    })

    return {
        "text": f"주제: {topic}",
        "attachments": atts,
        "responseType": response_type,          # A안 기본: "ephemeral"
        "replaceOriginal": replace_original      # 초기엔 False, 액션 갱신 시 True
    }

# -------- Core flow --------
def list_all_cards() -> List[str]:
    # public/card 아래 파일명을 URL 없이 리스트업
    card_dir = os.path.join(BASE_DIR, "public", "card")
    names = [f for f in os.listdir(card_dir) if f.lower().endswith(".jpg")]
    return sorted(names)

def stable_shuffle(all_names: List[str], seed: int) -> List[str]:
    r = random.Random(seed)
    names = all_names[:]
    r.shuffle(names)
    return names

# -------- Actions handler --------
async def handle_actions(req: Request, data: dict):
    action_value = data.get("actionValue")
    original     = data.get("originalMessage", {}) or {}
    topic        = (original.get("text") or "전반운").strip()

    def parse_state(v: str): return v.split("|")

    if action_value.startswith("pick|"):
        _, count_s, seed_s, picked_csv, choose_s = parse_state(action_value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        choose = int(choose_s)
        if choose not in picked: picked.append(choose)

        # 완료 시 결과 산출
        if len(picked) >= count:
            names = list_all_cards()
            deck  = stable_shuffle(names, seed)
            chosen_cards: List[Dict[str, Any]] = []
            for pos in picked:
                idx = pos - 1
                if 0 <= idx < len(deck):
                    chosen_cards.append({
                        "name": deck[idx],
                        "reversed": random.choice([True, False])
                    })

            reading = gpt_card_reading(chosen_cards, topic)

            atts: List[Dict[str, Any]] = []
            # 카드 이미지
            for c in chosen_cards:
                title = f"{c['name'].replace('.jpg','')} {'(역방향)' if c['reversed'] else '(정방향)'}"
                atts.append({
                    "title": title,
                    "imageUrl": public_url(req, f"/card/{c['name']}")
                })

            # 카드 해석 fields
            items = reading.get("items") or []
            if items:
                fields: List[Dict[str, Any]] = []
                for item in items:
                    fields.append({
                        "title": f"🔮 {item.get('name','')}",
                        "value": f"{item.get('position','')} | {item.get('keyword','')}\n👉 {item.get('meaning','')}\n💡 {item.get('advice','')}",
                        "short": False
                    })
                atts.append({"fields": fields})

            # 전체 요약
            summary = reading.get("summary")
            if summary:
                atts.append({"text": summary})

            return make_message(
                text="타로 결과",
                attachments=atts,
                response_type="ephemeral",      # 최초와 동일 스코프로 유지
                replace_original=True           # 기존 메시지 교체
            )

        # 아직 덜 골랐으면 선택 UI 갱신 (ephemeral 그대로)
        return build_pick_ui(
            req, count=count, picked=picked, seed=seed, topic=topic,
            response_type="ephemeral", replace_original=True
        )

    if action_value.startswith("reset|"):
        _, count_s, seed_s, picked_csv, topic2 = parse_state(action_value)
        return build_pick_ui(
            req, count=int(count_s), picked=[], seed=int(seed_s), topic=(topic2 or topic),
            response_type="ephemeral", replace_original=True
        )

    if action_value.startswith("fill|"):
        _, count_s, seed_s, picked_csv, topic2 = parse_state(action_value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        remain = [i for i in range(1, count+1) if i not in picked]
        random.shuffle(remain)
        picked += remain
        # 완료 루틴 재사용 (새 data 구성만 해서 재호출)
        return await handle_actions(req, {
            "actionValue": f"pick|{count}|{seed}|{','.join(map(str,picked[:-1]))}|{picked[-1]}",
            "originalMessage": {"text": topic2 or topic}
        })

    return {"text":"지원하지 않는 액션입니다.", "responseType":"ephemeral"}

# -------- Entry (single endpoint) --------
class SlashPayload(BaseModel):
    text: str | None = None

def verify_request(req: Request):
    # 필요시 헤더/토큰 검증
    expected = os.getenv("DOORAY_VERIFY_TOKEN")
    if not expected:
        return
    got = req.headers.get("X-Dooray-Token") or req.headers.get("Authorization")
    if got != expected:
        raise HTTPException(status_code=401, detail="invalid token")
@app.post("/dooray/command")
async def dooray_command(req: Request):
    verify_request(req)

    # (선택) 요청 원문도 찍고 싶으면 다음 2줄
    raw = (await req.body()).decode("utf-8", "ignore")
    logger.info("[IN] CT=%s RAW=%s", req.headers.get("content-type"), raw[:2000])

    data, is_action = await parse_dooray_payload(req)

    if is_action:  # 버튼 콜백
        payload = await handle_actions(req, data)   # dict를 반환한다고 가정
        return respond(payload, tag="action")

    # 슬래시 커맨드 → 스프레드 결정
    topic = (data.get("text") or "").strip() or "전반운"
    spread_info = decide_spread(topic)
    count = int(spread_info.get("card_count", 3))
    seed = random.randint(1, 2_000_000_000)

    payload = build_pick_ui(
        req, count=count, picked=[], seed=seed, topic=topic,
        response_type="ephemeral", replace_original=False
    )  # dict

    return respond(payload, tag="slash-init")



# ---------- 요청 로깅 미들웨어 ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        raw = await request.body()
    except Exception:
        raw = b""

    logger.info(
        "[IN] %s %s CT=%s UA=%s XFWD=%s RAW=%s",
        request.method,
        request.url.path,
        request.headers.get("content-type"),
        request.headers.get("user-agent"),
        request.headers.get("x-forwarded-for"),
        (raw.decode("utf-8", "ignore")[:2000]),  # 너무 길면 자르기
    )

    response = await call_next(request)

    # 응답의 상태/타입 정도는 여기서 찍고, 실제 payload는 respond()에서 찍음
    logger.info(
        "[OUT] %s %s -> %s (%s)",
        request.method,
        request.url.path,
        response.status_code,
        getattr(response, "media_type", None),
    )
    return response

# ---------- 응답 헬퍼 ----------
def respond(payload: Dict[str, Any], tag: str = "") -> JSONResponse:
    """
    Dooray로 보낼 payload를 stdout에 찍고 JSONResponse로 돌려준다.
    tag는 어디서 보낸 응답인지 구분용 라벨.
    """
    try:
        pretty = json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        pretty = f"<< payload json.dumps 실패: {e} >>"
    logger.info("[RESP%s] %s", f'/{tag}' if tag else "", pretty)
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

# --- 로컬 개발용 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
