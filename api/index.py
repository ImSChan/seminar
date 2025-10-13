from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple
from openai import OpenAI
import os, json, random, logging, sys

app = FastAPI(title="Dooray Tarot Bot")

# ---------------- Logging ----------------
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

# ---------------- OpenAI ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Data ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
KEYWORD_PATH = os.path.join(BASE_DIR, "data", "card_keywords.json")
with open(KEYWORD_PATH, "r", encoding="utf-8") as f:
    CARD_KEYWORDS: Dict[str, str] = json.load(f)

SPREAD_FILES = {1:"card_1.png",2:"card_2.png",3:"card_3.png",5:"card_5.png",6:"card_6.png",10:"card_10.png"}

# ---------------- Utils ----------------
def public_url(request: Request, path: str) -> str:
    base = os.getenv("APP_BASE_URL")
    if not base:
        scheme = request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        base = f"{scheme}://{host}"
    return f"{base}{path}"

async def parse_dooray_payload(req: Request) -> Tuple[Dict[str, Any], bool]:
    ctype = (req.headers.get("content-type") or "").lower()

    # JSON 우선
    if "application/json" in ctype:
        try:
            data = await req.json()
            is_action = bool(
                data.get("actionValue")
                or (data.get("actions") and data["actions"][0].get("value"))
            )
            if not data.get("actionValue") and data.get("actions"):
                data["actionValue"] = data["actions"][0].get("value")
                data["actionName"]  = data["actions"][0].get("name") or data.get("actionName")
            return data, is_action
        except Exception:
            pass

    # FORM 폴백
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
    return data, is_action

def make_message(text: str, attachments: List[Dict[str, Any]] = None,
                 response_type: str = "ephemeral",
                 replace_original: bool = False,
                 delete_original: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text": text,
        "responseType": response_type,
        "replaceOriginal": replace_original,
        "deleteOriginal": delete_original,
    }
    if attachments:
        payload["attachments"] = attachments
    return payload

def list_all_cards() -> List[str]:
    card_dir = os.path.join(BASE_DIR, "public", "card")
    names = [f for f in os.listdir(card_dir) if f.lower().endswith(".jpg")]
    return sorted(names)

def stable_shuffle(all_names: List[str], seed: int) -> List[str]:
    r = random.Random(seed)
    names = all_names[:]; r.shuffle(names); return names

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
{{
  "items":[{{"name":"","position":"","keyword":"","meaning":"","advice":""}}],
  "summary":"🧙 전체 해석: ..."
}}
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

# ---------------- UI builders ----------------
def build_confirm_ui(topic: str) -> Dict[str, Any]:
    # 투표 예제처럼: 슬래시 → 사용자에게만 보이는 확인창
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
                  response_type: str = "inChannel",
                  replace_original: bool = False,
                  delete_original: bool = False) -> dict:
    spread_file = SPREAD_FILES.get(count, SPREAD_FILES[3])
    spread_img_url = public_url(request, f"/card_spread/{spread_file}")
    picked_str = ", ".join(map(str, picked)) if picked else "없음"
    remain = [i for i in range(1, count + 1) if i not in picked]

    atts: List[Dict[str, Any]] = []
    atts.append({
        "callbackId": "tarot-pick",
        "text": f"🃏 번호를 순서대로 **{count}개** 선택해줘요.\n현재 선택: **{picked_str}**",
        "title": f"{count}장 스프레드",
        "imageUrl": spread_img_url
    })

    row: List[Dict[str, Any]] = []
    for i, num in enumerate(remain, start=1):
        row.append({"type":"button","text":str(num),"name":"pick",
                    "value":f"pick|{count}|{seed}|{','.join(map(str,picked))}|{num}",
                    "style":"default"})
        if i % 5 == 0:
            atts.append({"callbackId":"tarot-pick","actions":row}); row=[]
    if row:
        atts.append({"callbackId":"tarot-pick","actions":row})

    atts.append({
        "callbackId":"tarot-pick",
        "actions":[
            {"name":"reset","type":"button","text":"🔄 다시 선택",
             "value":f"reset|{count}|{seed}|{','.join(map(str,picked))}|{topic}"},
            {"name":"fill","type":"button","text":"🎲 무작위로 채우기",
             "value":f"fill|{count}|{seed}|{','.join(map(str,picked))}|{topic}"}
        ]
    })

    return {
        "text": f"주제: {topic}",
        "attachments": atts,
        "responseType": response_type,          # 채널 게시
        "replaceOriginal": replace_original,    # 업데이트 시 true
        "deleteOriginal": delete_original       # 최초 게시 시 true로 원본 삭제(확인창)
    }

def build_result_ui(req: Request, chosen_cards: List[Dict[str, Any]], reading: Dict[str, Any]) -> Dict[str, Any]:
    atts: List[Dict[str, Any]] = []
    for c in chosen_cards:
        title = f"{c['name'].replace('.jpg','')} {'(역방향)' if c['reversed'] else '(정방향)'}"
        atts.append({"title": title, "imageUrl": public_url(req, f"/card/{c['name']}")})

    items = reading.get("items") or []
    if items:
        fields = []
        for item in items:
            fields.append({
                "title": f"🔮 {item.get('name','')}",
                "value": f"{item.get('position','')} | {item.get('keyword','')}\n👉 {item.get('meaning','')}\n💡 {item.get('advice','')}",
                "short": False
            })
        atts.append({"fields": fields})

    summary = reading.get("summary")
    if summary:
        atts.append({"text": summary})

    return make_message(
        text="타로 결과",
        attachments=atts,
        response_type="inChannel",
        replace_original=True
    )

# ---------------- Verify ----------------
def verify_request(req: Request):
    expected = os.getenv("DOORAY_VERIFY_TOKEN")
    if not expected:
        return
    got = req.headers.get("X-Dooray-Token") or req.headers.get("Authorization")
    if got != expected:
        raise HTTPException(status_code=401, detail="invalid token")

# ---------------- Actions core ----------------
async def handle_actions_core(req: Request, data: dict) -> Dict[str, Any]:
    """인터랙티브 콜백 전용 처리 (별도 URL 또는 단일 URL 폴백 모두 이 로직 사용)"""
    action_name  = data.get("actionName")
    action_value = data.get("actionValue") or ""
    callback_id  = data.get("callbackId")
    original     = data.get("originalMessage", {}) or {}
    logger.info("[ACTION] name=%s value=%s cb=%s", action_name, action_value, callback_id)

    # 1) 리딩 시작/취소 (확인창 → 채널 게시)
    if callback_id == "tarot-confirm":
        if action_name == "cancel" or action_value == "cancel":
            return make_message("취소했어요.", response_type="ephemeral", replace_original=True)
        # 채널에 카드 선택 UI 새로 게시, 확인창은 삭제
        topic = (original.get("text") or action_value or "전반운").strip()
        spread = decide_spread(topic)
        count = int(spread.get("card_count", 3))
        seed  = random.randint(1, 2_000_000_000)
        return build_pick_ui(req, count=count, picked=[], seed=seed, topic=topic,
                             response_type="inChannel", replace_original=False, delete_original=True)

    # 2) 카드 선택/리셋/랜덤 채우기 (채널 메시지 업데이트)
    def parse_state(v: str): return v.split("|")

    if action_value.startswith("pick|"):
        _, count_s, seed_s, picked_csv, choose_s = parse_state(action_value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        choose = int(choose_s)
        if choose not in picked: picked.append(choose)

        if len(picked) >= count:
            names = list_all_cards()
            deck  = stable_shuffle(names, seed)
            chosen_cards = []
            for pos in picked:
                idx = pos - 1
                if 0 <= idx < len(deck):
                    chosen_cards.append({"name": deck[idx], "reversed": random.choice([True, False])})
            topic = (original.get("text") or "전반운").strip()
            reading = gpt_card_reading(chosen_cards, topic)
            return build_result_ui(req, chosen_cards, reading)

        topic = (original.get("text") or "전반운").strip()
        return build_pick_ui(req, count=count, picked=picked, seed=seed, topic=topic,
                             response_type="inChannel", replace_original=True)

    if action_value.startswith("reset|"):
        _, count_s, seed_s, picked_csv, topic = parse_state(action_value)
        return build_pick_ui(req, count=int(count_s), picked=[], seed=int(seed_s), topic=topic,
                             response_type="inChannel", replace_original=True)

    if action_value.startswith("fill|"):
        _, count_s, seed_s, picked_csv, topic = parse_state(action_value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        remain = [i for i in range(1, count+1) if i not in picked]
        random.shuffle(remain)
        picked += remain
        # 완료 루틴 재사용
        return await handle_actions_core(req, {
            "actionName": "pick",
            "actionValue": f"pick|{count}|{seed}|{','.join(map(str,picked[:-1]))}|{picked[-1]}",
            "originalMessage": original
        })

    return make_message("지원하지 않는 액션입니다.", response_type="ephemeral")

# ---------------- Endpoints ----------------
@app.post("/dooray/command")
async def dooray_command(req: Request):
    verify_request(req)
    raw = (await req.body()).decode("utf-8","ignore")
    logger.info("[IN] POST /dooray/command CT=%s RAW=%s", req.headers.get("content-type"), raw[:2000])

    data, is_action = await parse_dooray_payload(req)

    # 폴백: 어떤 테넌트는 인터랙티브도 같은 URL로 옴
    if is_action:
        payload = await handle_actions_core(req, data)
        return respond(payload, tag="action@command")

    # 슬래시 → ephemeral 확인창
    topic = (data.get("text") or "").strip() or "전반운"
    payload = build_confirm_ui(topic)
    return respond(payload, tag="slash-confirm")

@app.post("/dooray/actions")
async def dooray_actions(req: Request):
    # (가능하면 Dooray의 Interactive Message Request URL로 이 엔드포인트를 등록)
    verify_request(req)
    raw = (await req.body()).decode("utf-8","ignore")
    logger.info("[IN] POST /dooray/actions CT=%s RAW=%s", req.headers.get("content-type"), raw[:2000])

    data, _ = await parse_dooray_payload(req)
    payload = await handle_actions_core(req, data)
    return respond(payload, tag="action@actions")

# 로컬 테스트
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
