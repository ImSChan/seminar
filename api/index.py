from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
from openai import OpenAI
import os, json, random

app = FastAPI(title="Dooray Tarot Bot")

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

def attachment_text_block(text: str) -> Dict[str, Any]:
    return {"text": text}

def attachment_image_block(title: str, image_url: str, thumb_url: str = None,
                           author_name: str = None, title_link: str = None,
                           callback_id: str = None) -> Dict[str, Any]:
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

# -------- GPT Helpers --------
SPREAD_FILES = {
    1: "card_1.png",
    3: "card_3.png",
    5: "card_5.png",
    6: "card_6.png",
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
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":system_prompt.strip()},
            {"role":"user","content":topic}
        ],
        temperature=0.7,
    )
    text = res.choices[0].message.content.strip()
    # 안전하게 eval 대신 json.loads 시도
    try:
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
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt.strip()}],
        temperature=0.8,
    )
    txt = res.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except Exception:
        # 파싱 실패 시 통째로 텍스트를 summary로
        return {"items": [], "summary": txt}

def build_pick_ui(request: Request, count: int, picked: list[int], seed: int, topic: str) -> Dict[str, Any]:
    """
    번호 선택 UI를 attachments로 구성.
    버튼 value 포맷: pick|<count>|<seed>|<picked_csv>|<choose>
    """
    # 스프레드 이미지
    spread_file = SPREAD_FILES.get(count, SPREAD_FILES[3])
    spread_img_url = public_url(request, f"/card_spread/{spread_file}")

    # 현재 선택 현황 문구
    picked_str = ", ".join(map(str, picked)) if picked else "없음"
    remain = [i for i in range(1, count + 1) if i not in picked]

    atts = []
    # 안내 + 스프레드
    atts.append(attachment_text_block(f"🃏 번호를 순서대로 **{count}개** 선택해줘요.\n현재 선택: **{picked_str}**"))
    atts.append(attachment_image_block(title=f"{count}장 스프레드", image_url=spread_img_url))

    # 버튼은 한 줄에 너무 많으면 보기 안 좋으니 5개씩 끊어서 액션 블록 생성
    row = []
    for i, num in enumerate(remain, start=1):
        row.append(action_button(
            text=str(num),
            name="pick",
            value=f"pick|{count}|{seed}|{','.join(map(str,picked))}|{num}",
            style="default"
        ))
        if i % 5 == 0:
            atts.append(attachment_actions_block(row))
            row = []
    if row:
        atts.append(attachment_actions_block(row))

    # 리셋/랜덤
    atts.append(attachment_actions_block([
        action_button("🔄 다시 선택", "reset", f"reset|{count}|{seed}|{','.join(map(str,picked))}|{topic}", "default"),
        action_button("🎲 무작위로 채우기", "fill", f"fill|{count}|{seed}|{','.join(map(str,picked))}|{topic}", "default"),
    ]))

    return make_message(
        text=f"주제: {topic}",
        attachments=atts,
        response_type="inChannel",
        replace_original=True
    )


# -------- Core flow --------
def pick_random_cards(all_card_names: List[str], count: int) -> List[Dict[str, Any]]:
    random.shuffle(all_card_names)
    chosen = all_card_names[:count]
    return [{"name": n, "reversed": random.choice([True, False])} for n in chosen]

def list_all_cards() -> List[str]:
    # public/card 아래 파일명을 URL 없이 리스트업
    card_dir = os.path.join(BASE_DIR, "public", "card")
    names = [f for f in os.listdir(card_dir) if f.lower().endswith(".jpg")]
    return sorted(names)

# -------- Dooray Slash Command --------
class SlashPayload(BaseModel):
    # Dooray가 보내는 실제 필드명은 조직 설정에 따라 다를 수 있음.
    # 최소한 text, userId, channelId 같은 것만 참고.
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
    body = await req.json()
    topic = (body.get("text") or "").strip() or "전반운"

    # 1) 스프레드 결정
    spread_info = decide_spread(topic)
    count = int(spread_info.get("card_count", 3))
    reason = spread_info.get("reason", "해석을 위해")

    # 2) 덱 셔플을 위한 seed 생성(상태 없이 재현 가능)
    seed = random.randint(1, 2_000_000_000)

    # 3) 첫 화면: 이유 + 선택 UI
    intro = make_message(
        text=f"주제: {topic}",
        attachments=[
            attachment_text_block(f"🧐 {reason} → **{count}장**으로 볼게요!")
        ],
        response_type="inChannel",
        replace_original=False
    )

    # Dooray는 하나의 응답만 받는다면, 첫 화면 대신 곧바로 선택 UI만 보내도 OK.
    # 여기서는 선택 UI만 보내도록 바로 리턴:
    return build_pick_ui(req, count=count, picked=[], seed=seed, topic=topic)

def stable_shuffle(all_names: list[str], seed: int) -> list[str]:
    r = random.Random(seed)
    names = all_names[:]
    r.shuffle(names)
    return names

# -------- Dooray Interactive Actions --------

@app.post("/dooray/actions")
async def dooray_actions(req: Request):
    verify_request(req)
    data = await req.json()
    actions = data.get("actions", [])
    if not actions:
        raise HTTPException(status_code=400, detail="invalid action payload")

    value = actions[0].get("value", "")
    # originalMessage.text에 주제가 없을 수도 있으니 value에 topic을 담는 방식을 병행
    orig_text = data.get("originalMessage", {}).get("text", "").strip() or "전반운"

    def parse_state(v: str):
        # "pick|<count>|<seed>|<picked_csv>|<choose>"
        # "reset|<count>|<seed>|<picked_csv>|<topic>"
        # "fill|<count>|<seed>|<picked_csv>|<topic>"
        parts = v.split("|")
        return parts

    # ----- 번호 선택 -----
    if value.startswith("pick|"):
        _, count_s, seed_s, picked_csv, choose_s = parse_state(value)
        count = int(count_s)
        seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        choose = int(choose_s)

        if choose not in picked:
            picked.append(choose)

        # 다 고르면 결과 산출
        if len(picked) >= count:
            names = list_all_cards()                          # 파일명 목록
            deck = stable_shuffle(names, seed)               # seed 기반 결정적 셔플
            chosen_cards = []
            for pos in picked:
                idx = pos - 1
                if idx < 0 or idx >= len(deck):
                    continue
                chosen_cards.append({"name": deck[idx], "reversed": random.choice([True, False])})

            reading = gpt_card_reading(chosen_cards, orig_text)

            # 결과 attachments 구성
            atts = []
            for c in chosen_cards:
                title = f"{c['name'].replace('.jpg','')} {'(역방향)' if c['reversed'] else '(정방향)'}"
                img_url = public_url(req, f"/card/{c['name']}")
                atts.append(attachment_image_block(title=title, image_url=img_url))
            if reading.get("items"):
                fields = []
                for item in reading["items"]:
                    fields.append({
                        "title": f"🔮 {item.get('name','')}",
                        "value": f"{item.get('position','')} | {item.get('keyword','')}\n👉 {item.get('meaning','')}\n💡 {item.get('advice','')}",
                        "short": False
                    })
                atts.append(attachment_fields_block(fields))
            if reading.get("summary"):
                atts.append(attachment_text_block(reading["summary"]))

            return make_message(
                text="타로 결과",
                attachments=atts,
                response_type="inChannel",
                replace_original=True
            )

        # 아직 덜 골랐다면 선택 UI 갱신
        return build_pick_ui(req, count=count, picked=picked, seed=seed, topic=orig_text)

    # ----- 다시 선택 -----
    if value.startswith("reset|"):
        _, count_s, seed_s, picked_csv, topic = parse_state(value)
        count = int(count_s); seed = int(seed_s)
        return build_pick_ui(req, count=count, picked=[], seed=seed, topic=topic or orig_text)

    # ----- 무작위로 나머지 채우기 -----
    if value.startswith("fill|"):
        _, count_s, seed_s, picked_csv, topic = parse_state(value)
        count = int(count_s); seed = int(seed_s)
        picked = [int(x) for x in picked_csv.split(",") if x] if picked_csv else []
        remain = [i for i in range(1, count+1) if i not in picked]
        r = random.Random()  # 굳이 seed 고정 안 함(사용자 체감 ‘랜덤’)
        r.shuffle(remain)
        picked += remain
        # pick 완료와 동일 루틴 재사용
        fake_value = f"pick|{count}|{seed}|{','.join(map(str,picked[:-1]))}|{picked[-1]}"
        data["actions"][0]["value"] = fake_value
        return await dooray_actions(req)

    # (선택) 기존 draw_random 지원
    if value.startswith("draw_random:"):
        count = int(value.split(":")[1])
        seed = random.randint(1, 2_000_000_000)
        return build_pick_ui(req, count=count, picked=[], seed=seed, topic=orig_text)

    return make_message(text="지원하지 않는 액션이에요.", response_type="ephemeral")

# --- 로컬 개발용 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
