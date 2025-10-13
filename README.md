# Tarot Backend (FastAPI on Vercel)

간단한 FastAPI 백엔드. 로컬 실행 또는 Vercel 서버리스 배포 가능.

## 로컬 실행
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# uvicorn dev server
uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
