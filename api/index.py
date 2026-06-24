from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Tuple
from openai import OpenAI
import os, json, random, logging, sys

app = FastAPI(title="Dooray Tarot Bot")

# ---------- Logging ----------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(levelname)s %(asctime)s %(name)s : %(message)s",
)
logger = logging.getLogger("dooray-tarot")

def respond(payload: Dict[str, Any], tag: str = "") -> JSONResponse:
    try:
        logger.info("[RESP%s] %s", f'/{tag}' if tag else "", json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")

# ---------- OpenAI ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY is not set. GPT 요약은 기본값으로 동작합니다.")

# ---------- Data ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KEYWORD_PATH = os.path.join(BASE_DIR, "data", "card_keywords.json")
try:
    with open(KEYWORD_PATH, "r", encoding="utf-8") as f:
        CARD_KEYWORDS: Dict[str, str] = json.load(f)
except Exception as e:
    logger.warning("CARD_KEYWORDS load failed: %s", e)
    CARD_KEYWORDS = {}

SPREAD_FILES = {1:"card_1.png",2:"card_2.png",3:"card_3.png",5:"card_5.png",6:"card_6.png",10:"card_10.png"}

# ---------- Utils ----------
def public_url(request: Request, path: str) -> str:
    base = os.getenv("APP_BASE_URL")
    if not base:
        scheme = request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        base = f"{scheme}://{host}"
    return f"{base}{path}"

def extract_topic(t: str | None, default="전반운") -> str:
    if not t:
        return default
    s = t.strip()
    return s.split(":", 1)[1].strip() if s.startswith("주제:") else s

async def parse_dooray_payload(req: Request) -> Tuple[Dict[str, Any], bool]:
    ctype = (req.headers.get("content-type") or "").lower()
    # JSON 우선
    if "application/json" in ctype:
        try:
            data = await req.json()
            is_action = bool(
                data.get("actionValue") or (data.get("actions") and data["actions"][0].get("value"))
            )
            if not data.get("actionValue") and data.get("actions"):
                data["actionValue"] = data["actions"][0].get("value")
                data["actionName"]  = data["actions"][0].get("name") or data.get("actionName")
            logger.info("[PARSE] is_action=%s parsed=%s", is_action, json.dumps(data, ensure_ascii=False)[:2000])
            return data, is_action
        except Exception:
            pass
    # FORM 폼
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
                    data[k] = json.loads(v); continue
                except Exception:
                    pass
            data[k] = v
    if data.get("actions") and isinstance(data["actions"], list) and data["actions"]:
        if not data.get("actionValue"):
            v = data["actions"][0].get("value")
            if v: data["actionValue"] = v
        if not data.get("actionName"):
            n = data["actions"][0].get("name")
            if n: data["actionName"] = n
    is_action = bool(data.get("actionValue"))
    logger.info("[PARSE] is_action=%s parsed=%s", is_action, json.dumps(data, ensure_ascii=False)[:2000])
    return data, is_action

def make_message(text: str, attachments=None, response_type="ephemeral",
                 replace_original=False, delete_original=False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text": text,
        "responseType": response_type,
        "replaceOriginal": replace_original,
        "deleteOriginal": delete_original,
    }
    if attachments:
        payload["attachments"] = attachments
    return payload

# 변경
def list_all_cards() -> List[str]:
    # 정적 자산은 Vercel 함수 FS에서 안 보일 수 있으므로,
    # 데이터(키워드 파일) 키를 신뢰 소스로 사용
    names = [k for k in CARD_KEYWORDS.keys() if k.lower().endswith(".jpg")]
    names.sort()
    logger.info("[ASSET] cards via keywords: %d", len(names))
    return names

def stable_shuffle(all_names: List[str], seed: int) -> List[str]:
    r = random.Random(seed); names = all_names[:]; r.shuffle(names); return names

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
            messages=[{"role":"system","content":system_prompt.strip()},
                      {"role":"user","content":topic}],
            temperature=0.7,
        )
        return json.loads(res.choices[0].message.content.strip())
    except Exception:
        return {"spread":"3장","reason":"기본값으로 과거-현재-미래 흐름 확인","card_count":3}

def gpt_card_reading(cards: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
    cards_input = []
    for c in cards:
        name = c["name"].replace(".jpg", "")
        keyword = CARD_KEYWORDS.get(c["name"], "키워드 없음")
        position = "역방향" if c["reversed"] else "정방향"
        cards_input.append({"name": name, "position": position, "keyword": keyword})
    prompt = f"""
당신은 숙련된 타로카드 리더입니다.
'{topic}' 주제로 사용자가 아래 카드를 뽑았습니다:
{json.dumps(cards_input, ensure_ascii=False, indent=2)}
JSON만 반환(코드블록 금지):
{{"items":[{{"name":"","position":"","keyword":"","meaning":"","advice":""}}],"summary":"🧙 전체 해석: ..."}}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt.strip()}],
            temperature=0.8,
        )
        return json.loads(res.choices[0].message.content.strip())
    except Exception:
        return {"items": [], "summary": "🧙 전체 해석: 핵심에 집중하여 균형 있게 결정하세요."}

# ---------- UI Builders ----------
def build_confirm_ui(topic: str) -> Dict[str, Any]:
    return make_message(
        text="리딩을 시작할까요?",
        attachments=[
            {
                "title": f"주제: {topic}",
                "fields": [{"title":"설명","value":"'리딩 시작'을 누르면 채널에 카드 선택 UI가 게시됩니다.","short":False}],
            },
            {
                "callbackId":"tarot-confirm",
                "actions":[
                    {"name":"start","type":"button","text":"리딩 시작","value":topic,"style":"primary"},
                    {"name":"cancel","type":"button","text":"취소","value":"cancel"}
                ]
            }
        ],
        response_type="ephemeral",
        replace_original=False
    )

def build_pick_ui(request: Request, count: int, picked: List[int], seed: int, topic: str,
                  response_type="ephemeral", replace_original=False, delete_original=False) -> dict:
    spread_file = SPREAD_FILES.get(count, SPREAD_FILES[3])
    spread_img_url = public_url(request, f"/card_spread/{spread_file}")
    picked_str = ", ".join(map(str, picked)) if picked else "없음"
    remain = [i for i in range(1, count + 1) if i not in picked]

    atts: List[Dict[str, Any]] = []
    atts.append({"callbackId":"tarot-pick","text":f"🃏 번호를 순서대로 **{count}개** 선택해줘요.\n현재 선택: **{picked_str}**",
                 "title":f"{count}장 스프레드","imageUrl":spread_img_url})

    row: List[Dict[str, Any]] = []
    for i, num in enumerate(remain, start=1):
        row.append({"type":"button","text":str(num),"name":"pick",
                    "value":f"pick|{count}|{seed}|{','.join(map(str,picked))}|{num}",
                    "style":"default"})
        if i % 5 == 0:
            atts.append({"callbackId":"tarot-pick","actions":row}); row=[]
    if row: atts.append({"callbackId":"tarot-pick","actions":row})

    atts.append({"callbackId":"tarot-pick","actions":[
        {"name":"reset","type":"button","text":"🔄 다시 선택",
         "value":f"reset|{count}|{seed}|{','.join(map(str,picked))}|{topic}"},
        {"name":"fill","type":"button","text":"🎲 무작위로 채우기",
         "value":f"fill|{count}|{seed}|{','.join(map(str,picked))}|{topic}"}
    ]})

    return {"text": f"주제: {topic}",
            "attachments": atts,
            "responseType": response_type,
            "replaceOriginal": replace_original,
            "deleteOriginal": delete_original}

from urllib.parse import quote
def build_result_ui(req: Request, chosen_cards: List[Dict[str, Any]], reading: Dict[str, Any]) -> Dict[str, Any]:
    atts: List[Dict[str, Any]] = []
    for c in chosen_cards:
        title = f"{c['name'].replace('.jpg','')} {'(역방향)' if c['reversed'] else '(정방향)'}"
        # 파일명 인코딩
        img_url = public_url(req, f"/card/{quote(c['name'])}")
        atts.append({"title": title, "imageUrl": img_url})
        
    items = reading.get("items") or []
    if items:
        fields = []
        for item in items:
            fields.append({"title": f"🔮 {item.get('name','')}",
                           "value": f"{item.get('position','')} | {item.get('keyword','')}\n👉 {item.get('meaning','')}\n💡 {item.get('advice','')}",
                           "short": False})
        atts.append({"fields": fields})
    summary = reading.get("summary")
    if summary: atts.append({"text": summary})
    return make_message(text="타로 결과", attachments=atts, response_type="ephemeral", replace_original=True)

# ---------- Verify ----------
def verify_request(req: Request):
    expected = os.getenv("DOORAY_VERIFY_TOKEN")
    if not expected: return
    got = req.headers.get("X-Dooray-Token") or req.headers.get("Authorization")
    if got != expected:
        raise HTTPException(status_code=401, detail="invalid token")

# ---------- Actions core ----------
async def handle_actions_core(req: Request, data: dict) -> Dict[str, Any]:
    action_name  = data.get("actionName")
    action_value = (data.get("actionValue") or "").strip()
    callback_id  = data.get("callbackId")
    original     = data.get("originalMessage", {}) or {}
    logger.info("[ACTION] name=%s value=%s cb=%s", action_name, action_value, callback_id)

    # 1) 확인창 단계
    if callback_id == "tarot-confirm":
        if action_name == "cancel" or action_value == "cancel":
            return make_message("취소했어요.", response_type="ephemeral", replace_original=True)
        # actionValue(버튼에 실린 실제 주제) 우선, 없으면 originalMessage에서 추출
        topic = (action_value or extract_topic(original.get("text"), "전반운"))
        spread = decide_spread(topic)
        count = int(spread.get("card_count", 3))
        seed  = random.randint(1, 2_000_000_000)
        # 채널에 새로 게시 + 확인창 삭제
        return build_pick_ui(req, count=count, picked=[], seed=seed, topic=topic,
                             response_type="ephemeral", replace_original=False, delete_original=True)

    # 2) 카드 선택 단계
    def parse_state(v: str): return v.split("|")

    if action_value.startswith("pick|"):
        _, count_s, seed_s, picked_csv, choose_s = parse_state(action_value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        choose = int(choose_s)
        if choose not in picked: picked.append(choose)

        # 선택 완료
        if len(picked) >= count:
            names = list_all_cards()
            logger.info("[ASSET] cards=%d", len(names))
            if not names:
                # 에셋 문제 안내 (채널에 공지)
                return make_message(
                    text="⚠️ 카드 이미지가 없습니다. 저장소의 /public/card 에 .jpg 파일을 넣어주세요.",
                    response_type="ephemeral", replace_original=False
                )

            deck = stable_shuffle(names, seed)
            logger.info("[DECK] seed=%s first5=%s", seed, deck[:5])

            chosen_cards = []
            for pos in picked:
                idx = pos - 1
                if 0 <= idx < len(deck):
                    chosen_cards.append({"name": deck[idx], "reversed": random.choice([True, False])})
                else:
                    logger.error("[INDEX] pos=%s out of range deck_len=%s", pos, len(deck))

            if not chosen_cards:
                return make_message(
                    text="⚠️ 선택한 번호가 덱 범위를 벗어났어요. 다시 시도해 주세요.",
                    response_type="ephemeral", replace_original=False
                )

            topic = extract_topic(original.get("text"), "전반운")
            reading = gpt_card_reading(chosen_cards, topic)

            return build_result_ui(req, chosen_cards, reading)

        # 아직 선택 미완료 → UI 갱신
        topic = extract_topic(original.get("text"), "전반운")
        return build_pick_ui(req, count=count, picked=picked, seed=seed, topic=topic,
                            response_type="ephemeral", replace_original=True)

    if action_value.startswith("reset|"):
        _, count_s, seed_s, picked_csv, topic2 = parse_state(action_value)
        topic = topic2 or extract_topic(original.get("text"), "전반운")
        return build_pick_ui(req, count=int(count_s), picked=[], seed=int(seed_s), topic=topic,
                            response_type="ephemeral", replace_original=True)
    
    if action_value.startswith("fill|"):
        _, count_s, seed_s, picked_csv, topic2 = parse_state(action_value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        remain = [i for i in range(1, count+1) if i not in picked]
        random.shuffle(remain); picked += remain
        # 완료 루틴 재사용 (topic은 original에서 복구되거나 이후 단계에서 extract_topic 사용)
        return await handle_actions_core(req, {
            "actionName": "pick",
            "actionValue": f"pick|{count}|{seed}|{','.join(map(str,picked[:-1]))}|{picked[-1]}",
            "originalMessage": {"text": f"주제: {topic2 or extract_topic(original.get('text'), '전반운')}"}
        })

    return make_message("지원하지 않는 액션입니다.", response_type="ephemeral")

# ---------- Endpoints ----------
@app.post("/dooray/command")
async def dooray_command(req: Request):
    verify_request(req)
    raw = (await req.body()).decode("utf-8","ignore")
    logger.info("[IN] POST /dooray/command CT=%s RAW=%s", req.headers.get("content-type"), raw[:2000])
    data, is_action = await parse_dooray_payload(req)

    # 어떤 테넌트는 액션도 같은 URL로 오므로 폴백 허용
    if is_action:
        return respond(await handle_actions_core(req, data), tag="action@command")

    topic = (data.get("text") or "").strip() or "전반운"
    return respond(build_confirm_ui(topic), tag="slash-confirm")
@app.post("/dooray/gpt")
async def dooray_gpt(req: Request):
    verify_request(req)

    raw = (await req.body()).decode("utf-8", "ignore")
    logger.info("[IN] POST /dooray/gpt CT=%s RAW=%s", req.headers.get("content-type"), raw[:2000])

    data, is_action = await parse_dooray_payload(req)

    question = (
        data.get("text")
        or data.get("question")
        or data.get("actionValue")
        or ""
    ).strip() + "**반드시 한국어로 대답!**"

    if not question:
        return respond(
            make_message(
                text="질문 내용을 입력해 주세요.",
                response_type="ephemeral"
            ),
            tag="dooray-gpt-empty"
        )

    try:
        res = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.4"),
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.7,
        )

        answer = res.choices[0].message.content or ""

        return respond(
            make_message(
                text=answer,
                response_type="ephemeral",
                replace_original=False
            ),
            tag="dooray-gpt"
        )

    except Exception as e:
        logger.exception("[DOORAY_GPT] failed: %s", e)

        return respond(
            make_message(
                text="⚠️ GPT 질의 중 오류가 발생했어요. 로그를 확인해 주세요.",
                response_type="ephemeral"
            ),
            tag="dooray-gpt-error"
        )
@app.post("/dooray/actions")
async def dooray_actions(req: Request):
    verify_request(req)
    raw = (await req.body()).decode("utf-8","ignore")
    logger.info("[IN] POST /dooray/actions CT=%s RAW=%s", req.headers.get("content-type"), raw[:2000])
    try:
        data, _ = await parse_dooray_payload(req)
        payload = await handle_actions_core(req, data)
        return respond(payload, tag="action@actions")
    except Exception as e:
        logger.exception("[UNHANDLED] actions crashed: %s", e)
        # Dooray는 500을 싫어함. 200으로 에러 메시지 반환
        return respond(make_message(
            text="⚠️ 내부 오류가 발생했어요. 로그를 확인 중입니다.",
            response_type="ephemeral"
        ), tag="action@actions-error")

# ----- Local run -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
