import os
import token
from typing import List, Optional, Any, Dict
from urllib import response
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from typing import List, Optional, Any, Dict
from pathlib import Path
from pydantic import BaseModel
from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
    FaissManager,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis
from datetime import timedelta

# redis_client = redis.Redis(host="localhost", port=8080, db=0, decode_responses=True)
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def cache_token(token: str, username: str, expires_in: int = 900):
    redis_client.setex(f"token:{token}", timedelta(seconds=expires_in), value=username)
    
    
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

app = FastAPI(title="Document Portal API", version="0.1")

# serve static & templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str( BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080"],  # or whatever your frontend runs on
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET_KEY = "My-personal-secret-key"
ALGORITHM = "HS256"

def get_current_user_from_token(token: str) -> str:
    cached_user = redis_client.get(f"token:{token}")
    if cached_user:
        return cached_user

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username:
            cache_token(token, username)
            return username
    except JWTError:
        pass

    raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    cached_user = redis_client.get(f"token:{token}")
    if cached_user:
        return cached_user  # Token is valid and cached

    # Fallback to JWT decode if not cached
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username:
            cache_token(token, username)
            return username
    except JWTError:
        pass

    raise HTTPException(status_code=401, detail="Invalid token")

def get_token_from_cookie(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token

@app.get("/",response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse("login.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}
#-------------------Login API------------------------------------
class LoginData(BaseModel):
    username: str
    password: str
    
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



@app.post("/login")
async def login(data: LoginData):
    print("Login route hit")
    if data.username == "admin" and data.password == "secret":
        token_data = {"sub": data.username}
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)       
        # cache_token(token, data.username)
         # Optionally set token in cookie

        response = JSONResponse(content={"message": "Login successful"})
        # response.set_cookie(key="access_token", value=token, httponly=True)
        return response

        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(key="access_token", value=token, httponly=True)
        return response
       
    return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})
# @app.post("/login")
# async def login(data: LoginData):
#     try:
#         print("Login route hit")
#         # your logic here
#         return JSONResponse(content={"message": "Login successful"})
#     except Exception as e:
#         print("Error:", e)
#         return JSONResponse(status_code=500, content={"detail": "Internal error"})
    
@app.post("/logout")
async def logout(request: Request):
    # Replace with real logout logic
    resp = templates.TemplateResponse("login.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp


#-------------------Dashboard API------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: str):
    resp = templates.TemplateResponse("index.html", {"request": request, "user": user})
    resp.headers["Cache-Control"] = "no-store"
    return resp
# @app.get("/dashboard", response_class=HTMLResponse , token: str = Depends(get_token_from_cookie))
# async def dashboard(request: Request):
#     user = get_current_user(token)  # reuse your JWT validation logic
#     resp = templates.TemplateResponse("index.html", {"request": request, "user": user})
#     resp.headers["Cache-Control"] = "no-store"
#     return resp

# @app.get("/dashboard", response_class=HTMLResponse)
# async def dashboard(request: Request):
#     token = get_token_from_cookie(request)  # ✅ Extract token from cookie
#     user = get_current_user_from_token(token)        # ✅ Validate token and get user
#     resp = templates.TemplateResponse("index.html", {"request": request, "user": user})
#     resp.headers["Cache-Control"] = "no-store"
#     return resp

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_file(FastAPIFileAdapter(file))
        text = _read_file_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        # deepeval = analyzer.find_deepEval(text,result)
        deepeval = {'metrics': {'T1' :'20', 'T2': '30'}, 'score' : '0.23'}
        response_data = {**result, **deepeval}
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])

        # deepEval = {metric: score for metric, score in rag.find_deepEval(rag.context, question, response).items()}
        deepEval = [{'metric': 2.4, 'score': 0.95}]  # Dummy result, replace with actual call
        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG",
            "deepEval": deepEval
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ---------- Helpers ----------
class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile -> .name + .getbuffer() API"""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename
    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

def _read_file_via_handler(handler: DocHandler, path: str) -> str:
    if hasattr(handler, "read_file"):
        return handler.read_file(path)  # type: ignore
    raise RuntimeError("DocHandler has no read_file method.")

# backend.py

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dummy metric result
@app.get("/api/evaluation")
def get_evaluation(context: str, prompt: str, answer: str):
    # Replace with actual DeepEval logic
    rag = ConversationalRAG(session_id="abc")
    rag.load_retriever_from_faiss(index_dir="faiss_index/abc", k=5, index_name="index")
    deep_eval_results = rag.find_deepEval(context, prompt,answer)
    return {"DeepEval": deep_eval_results}

# command for executing the fast api
# uvicorn api.main:app --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload