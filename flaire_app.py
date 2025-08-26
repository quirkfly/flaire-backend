import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import dotenv_values
from openai import OpenAI
import requests
import logging
import jwt
from sqlalchemy import create_engine, Column, String, DateTime, text
from sqlalchemy.orm import declarative_base, Session
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, '.env')  # expect project/.env (one level up)

if not os.path.exists(ENV_PATH):
    raise SystemExit(f".env file not found at {ENV_PATH}. Create it with OPENAI_API_KEY=<key>.")

env_values = dotenv_values(ENV_PATH)
OPENAI_API_KEY = env_values.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY missing from .env file.")

# Model name as requested. (Note: 'chatgpt-5' may not be a currently deployed model.)
OPENAI_MODEL = env_values.get("OPENAI_MODEL", "chatgpt-5")
DEBUG = env_values.get("OPENAI_DEBUG", "0") in ("1", "true", "TRUE", "yes", "YES")
JWT_SECRET = env_values.get("JWT_SECRET") or os.getenv("JWT_SECRET") or "dev-insecure-secret-change"
JWT_ALG = "HS256"
JWT_DAYS = int(env_values.get("JWT_DAYS", 7))

# Database URL (example: postgresql://user:pass@localhost:5432/flare)
DATABASE_URL = env_values.get("DATABASE_URL") or os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=False, future=True) if DATABASE_URL else None
Base = declarative_base()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("flaire_app")
if not DATABASE_URL:
    logger.warning("DATABASE_URL not set; signup persistence disabled.")

# Lazy client init so import doesn't crash if proxies kw mismatch occurs
_openai_client = None
_openai_client_error: Optional[str] = None

def get_openai_client():
    global _openai_client, _openai_client_error
    if _openai_client or _openai_client_error:
        return _openai_client
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        _openai_client_error = str(e)
    return _openai_client

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Flaire Profile Generator (Minimal)",
    version="0.1.0",
    description="Minimal backend focused on profile generation using OpenAI Chat API"
)

# ---------------------------------------------------------------------------
# CORS (needed for browser POST with application/json which triggers preflight)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
"""Backend service providing:
 - /auth/signup & /auth/signin (JWT issuance)
 - /conversation/generate (conversation starters)
 - /profile/generate-from-photos (profile generation from image data URLs)
 - /health, /debug/routes

 All legacy upload & intermediate analysis endpoints were removed for minimal surface area.
"""

# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class GenerateFromPhotosRequest(BaseModel):
    photo_urls: List[str]
    preferences: Optional[Dict[str, Any]] = None
class GenerateProfileResponse(BaseModel):
    bio: str
    traits: List[str]
    interests: List[str]
    match_percentage: int
    profile_strength: str
    model_used: str
    openai_latency_ms: Optional[float] = None
    diagnostics: Optional[Dict[str, Any]] = None

class ConversationStartersRequest(BaseModel):
    photo_urls: List[str]
    context: Optional[str] = None  # optional extra text (their bio snippet etc.)
    preferences: Optional[Dict[str, Any]] = None

class ConversationStartersResponse(BaseModel):
    starters: List[str]
    guidance: Optional[str] = None
    model_used: str
    openai_latency_ms: Optional[float] = None
    diagnostics: Optional[Dict[str, Any]] = None

class SignUpRequest(BaseModel):
    name: str
    email: str
    password: str

class SignUpResponse(BaseModel):
    id: str
    email: str
    name: str
    plan: str
    created_at: str
    token: str

class SignInRequest(BaseModel):
    email: str
    password: str

class SignInResponse(SignUpResponse):
    pass

# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    plan = Column(String, nullable=False, server_default=text("'free'"))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

if engine is not None:
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        logger.error("Failed creating tables: %s", e)

# Utility functions
def hash_password(pw: str) -> str:
    try:
        from passlib.hash import bcrypt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Password hashing unavailable: {e}")
    return bcrypt.hash(pw)

def verify_password(pw: str, hashed: str) -> bool:
    try:
        from passlib.hash import bcrypt
    except Exception:
        return False
    try:
        return bcrypt.verify(pw, hashed)
    except Exception:
        return False

def normalize_email(email: str) -> str:
    return email.strip().lower()

# ---------------------------------------------------------------------------
# Auth / Signup Endpoint
# ---------------------------------------------------------------------------
@app.post('/auth/signup', response_model=SignUpResponse)
def signup(req: SignUpRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Persistence unavailable (DATABASE_URL not configured)")
    email = normalize_email(req.email)
    if '@' not in email:
        raise HTTPException(status_code=400, detail="Invalid email")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password too short")
    with Session(engine) as session:
        existing = session.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(status_code=409, detail="Email already registered")
        user = User(email=email, name=req.name.strip() or email.split('@')[0], password_hash=hash_password(req.password), plan='free')
        session.add(user)
        session.commit()
        session.refresh(user)
    token = create_jwt(user)
    return SignUpResponse(id=user.id, email=user.email, name=user.name, plan=user.plan, created_at=user.created_at.isoformat(), token=token)

@app.post('/auth/signin', response_model=SignInResponse)
def signin(req: SignInRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Persistence unavailable (DATABASE_URL not configured)")
    email = normalize_email(req.email)
    with Session(engine) as session:
        user = session.query(User).filter(User.email == email).first()
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_jwt(user)
        return SignInResponse(id=user.id, email=user.email, name=user.name, plan=user.plan, created_at=user.created_at.isoformat(), token=token)

# ---------------------------------------------------------------------------
# JWT Helper
# ---------------------------------------------------------------------------
def create_jwt(user: 'User') -> str:
    now = int(time.time())
    payload = {
        'sub': user.id,
        'email': user.email,
        'plan': user.plan,
        'iat': now,
        'exp': now + JWT_DAYS * 86400
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a dating profile expert who crafts attractive, authentic, masculine profiles that increase match rates. "
    "Return ONLY valid JSON strictly matching the requested schema."
)

def default_profile() -> Dict[str, Any]:
    return {
        "bio": "Adventurous, growth-minded and always finding a reason to laugh. Builder, traveler and gym regular who values genuine connection over small talk.",
        "traits": ["Adventurous", "Driven", "Grounded", "Curious", "Confident"],
        "interests": ["Travel", "Fitness", "Cooking", "Outdoors", "Live Music"]
    }

def parse_response(text: str) -> Dict[str, Any]:
    # Attempt to robustly extract the first valid JSON object.
    if not text:
        return default_profile()
    # Quick scan for fenced code blocks
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            part = part.strip()
            if part.startswith('{') and part.endswith('}'):
                try:
                    obj = json.loads(part)
                    if isinstance(obj.get('traits'), list) and isinstance(obj.get('interests'), list):
                        return obj
                except Exception:
                    continue
    # Fallback: locate outermost braces and try progressively shorter endings
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        for trim in range(0, 50):  # try trimming trailing chars to salvage JSON
            try:
                obj = json.loads(candidate[:len(candidate)-trim] if trim else candidate)
                if isinstance(obj.get('traits'), list) and isinstance(obj.get('interests'), list):
                    return obj
            except Exception:
                continue
    return default_profile()

def assess_match_percentage(insights: Dict[str, Any]) -> int:
    score = 70
    if insights['avg_aesthetic_score'] > 0.7:
        score += 15
    elif insights['avg_aesthetic_score'] > 0.5:
        score += 8
    score += min(len(insights['activities']) * 2, 10)
    if insights['photo_count'] >= 4:
        score += 5
    return min(score, 95)

def profile_strength(match_pct: int) -> str:
    if match_pct >= 90: return "Outstanding"
    if match_pct >= 80: return "Excellent"
    if match_pct >= 70: return "Very Good"
    if match_pct >= 60: return "Good"
    return "Needs Improvement"

def chat_completion(messages: List[Dict[str, Any]], diagnostics: Dict[str, Any], temperature: float = 0.7, max_tokens: int = 800) -> Optional[str]:
    """Call OpenAI via SDK if possible, else fallback to raw HTTP REST."""
    logger.info("[CHAT_COMPLETION] messages=%s", messages)
  
    models_order = []
    seen = set()
    primary_chain = [OPENAI_MODEL, 'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
    for m in primary_chain:
        if m and m not in seen:
            seen.add(m)
            models_order.append(m)
    last_error = None
    sdk_available = True
    client = get_openai_client()
    if client is None:
        sdk_available = False
        diagnostics['sdk_init_error'] = _openai_client_error
    if sdk_available:
        for idx, model_name in enumerate(models_order, start=1):
            start = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                latency = (time.perf_counter() - start) * 1000
                diagnostics.setdefault('attempts', []).append({'model': model_name, 'success': True, 'latency_ms': round(latency,2)})
                diagnostics['model_used'] = model_name
                return resp.choices[0].message.content
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                last_error = e
                diagnostics.setdefault('attempts', []).append({'model': model_name, 'success': False, 'latency_ms': round(latency,2), 'error': str(e)[:200]})
                if DEBUG:
                    logger.warning("[OPENAI_ATTEMPT_FAIL] model=%s idx=%s error=%s", model_name, idx, e)
                continue
    # REST fallback if SDK failed or not usable
    if sdk_available and last_error:
        diagnostics['final_error'] = str(last_error)
    rest_start = time.perf_counter()
    try:
        # Use first model only for REST path
        rest_model = models_order[0]
        payload = {
            'model': rest_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers={
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }, json=payload, timeout=40)
        rest_latency = (time.perf_counter() - rest_start) * 1000
        diagnostics.setdefault('attempts', []).append({'model': rest_model, 'path': 'http', 'success': resp.status_code==200, 'latency_ms': round(rest_latency,2), 'status': resp.status_code})
        if resp.status_code != 200:
            diagnostics['http_error'] = resp.text[:300]
            return None
        data = resp.json()
        diagnostics['model_used'] = rest_model
        return data.get('choices',[{}])[0].get('message',{}).get('content')
    except Exception as e:
        diagnostics['rest_exception'] = str(e)
        return None

# ---------------------------------------------------------------------------
# Conversation Starters (multimodal) endpoint
# ---------------------------------------------------------------------------
def build_starters_responses_content(photo_urls: List[str], context: Optional[str], preferences: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_prompt = (
        "You are an expert at crafting high-conversion, respectful dating conversation starters. "
    "Given the photos (and optional context), produce 3 varied openers. Each opener must: "
        "1) Reference visible details or provided context, 2) Be under 220 characters, 3) Avoid generic compliments, "
        "4) Feel authentic, playful or curiosity-driving, 5) Avoid sensitive inferences (age, ethnicity, religion, profession)."
    )
    if preferences:
        pref_line = ", ".join(f"{k}: {v}" for k,v in preferences.items())
        base_prompt += f" Preferences: {pref_line}."
    if context:
        base_prompt += f" Context Provided: {context.strip()}"
    base_prompt += "\nReturn ONLY JSON: {\"starters\":[\"...\",...],\"guidance\":\"short strategic tip\"}."
    content: List[Dict[str, Any]] = [
        {"role": "user", "content": [
            {"type": "input_text", "text": base_prompt}
        ]}
    ]
    for url in photo_urls[:5]:  # limit to 5
        content[0]['content'].append({"type": "input_image", "image_url": url})
    return content

@app.post('/conversation/generate', response_model=ConversationStartersResponse)
async def generate_conversation_starters(req: ConversationStartersRequest = Body(...)):
    if not req.photo_urls:
        raise HTTPException(status_code=400, detail="photo_urls is empty")
    if len(req.photo_urls) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 photo URLs")
    diagnostics: Dict[str, Any] = {'mode': 'conversation', 'model': OPENAI_MODEL, 'correlation_id': uuid.uuid4().hex[:12]}
    client = get_openai_client()
    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client unavailable")
    content = build_starters_responses_content(req.photo_urls, req.context, req.preferences)
    start = time.perf_counter()
    starters_text = None
    try:
        if hasattr(client, 'responses'):
            resp = client.responses.create(model=OPENAI_MODEL, input=content)
            latency = (time.perf_counter() - start) * 1000
            diagnostics.setdefault('attempts', []).append({'api':'responses','latency_ms': round(latency,2),'success': True})
            diagnostics['model_used'] = OPENAI_MODEL
            starters_text = getattr(resp, 'output_text', None) or str(resp)
        else:
            diagnostics['responses_missing'] = True
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        diagnostics.setdefault('attempts', []).append({'api':'responses','latency_ms': round(latency,2),'success': False,'error': str(e)[:160]})
        logger.warning('[CONVO_RESPONSES_FAIL] %s', e)
    parsed: Dict[str, Any] = {}
    if starters_text:
        try:
            # Extract first JSON object
            lb = starters_text.find('{')
            rb = starters_text.rfind('}')
            if lb != -1 and rb != -1 and rb > lb:
                parsed = json.loads(starters_text[lb:rb+1])
        except Exception:
            parsed = {}
    starters = parsed.get('starters') if isinstance(parsed.get('starters'), list) else []
    if not starters:
        # fallback quick generation using chat completion without images (summary only)
        fallback_prompt = "Generate 3 creative, specific dating app opening messages (JSON list) based only on visible photo cues."
        cc_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fallback_prompt}
        ]
        cc_content = chat_completion(cc_messages, diagnostics)
        if cc_content:
            try:
                lb = cc_content.find('[')
                rb = cc_content.rfind(']')
                if lb != -1 and rb != -1:
                    starters = json.loads(cc_content[lb:rb+1])
            except Exception:
                pass
    if not starters:
        starters = [
            "Noticed the energy in your photos—what's one adventure that totally reset your perspective lately?",
            "Quick question: spontaneous sunrise mission or late-night city wander—what's more you?",
            "Your pics suggest good stories—what's the most unexpected moment behind one of them?",
            "If we borrowed the vibe of one photo for a first hang, what are we doing?",
            "Two truths and a wild plan: tell me one real thing you're into and pitch a fun micro-adventure.",
            "Serious debate: which photo is the most you—and why?"
        ]
    guidance = parsed.get('guidance') if isinstance(parsed.get('guidance'), str) else "Lead with specificity. Match their energy; be curious over impressed."
    resp = ConversationStartersResponse(
        # Ensure only 3 starters returned
        starters=starters[:3],
        guidance=guidance,
        model_used=diagnostics.get('model_used', OPENAI_MODEL),
        openai_latency_ms=(diagnostics.get('attempts', [{}])[-1].get('latency_ms') if diagnostics.get('attempts') else None),
        diagnostics=diagnostics if DEBUG else None
    )
    return resp

def build_photo_messages(photo_urls: List[str], preferences: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    instruction = (
        "You will analyze up to 5 dating profile photos and then produce ONLY JSON with a generated profile. "
        "For each image infer activities, style indicators, and aesthetic quality (0-1). "
        "Then craft the profile JSON as previously specified."
    )
    logger.info("[BUILD_PHOTO_MSG] photo_count=%d", len(photo_urls))
    logger.info("[BUILD_PHOTO_MSG] sample_urls=%s", photo_urls[:3])
    pref_text = ""
    if preferences:
        pref_text = "Preferences: " + ", ".join(f"{k}: {v}" for k,v in preferences.items())
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": instruction + ("\n"+pref_text if pref_text else "") }]
    # Always attach images directly (including data: URIs) so model can access visual content.
    sample_list = []
    for url in photo_urls:
        if url.startswith('data:'):
            # For logging: capture mime + short head of payload
            meta = url.split(',')[0]
            mime = meta[5:].split(';')[0] if meta.startswith('data:') else 'unknown'
            head = url.split(',')[1][:16] if ',' in url else ''
            sample_list.append(f"data:{mime} b64_head={head}...")
        else:
            sample_list.append(url[:60] + ('...' if len(url) > 60 else ''))
        user_content.append({"type": "image_url", "image_url": {"url": url}})
    if DEBUG:
        logger.debug("[PHOTO_MSG_BUILD] attached_images=%d samples=%s", len(photo_urls), sample_list[:3])
    final_prompt = (
        "Return ONLY JSON in this exact schema (no markdown, no extra text):\n" +
        '{"bio":"...","traits":["t1","t2","t3","t4","t5"],"interests":["i1","i2","i3","i4","i5"]}'
    )
    user_content.append({"type": "text", "text": final_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

# ---------------------------------------------------------------------------
@app.post('/profile/generate-from-photos', response_model=GenerateProfileResponse)
async def generate_profile_from_photos(req: GenerateFromPhotosRequest = Body(...)):
    if not req.photo_urls:
        raise HTTPException(status_code=400, detail="photo_urls is empty")
    if len(req.photo_urls) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 photo URLs")
    diagnostics: Dict[str, Any] = {'mode': 'photo_end_to_end', 'model': OPENAI_MODEL, 'correlation_id': uuid.uuid4().hex[:12]}
    # Single path: build messages with images then attempt chat completion (with internal fallback HTTP path)
    messages = build_photo_messages(req.photo_urls, req.preferences)
    cc_content = chat_completion(messages, diagnostics)
    parsed = parse_response(cc_content or "") if cc_content else default_profile()
    # If parse failed (default bio), attempt stricter retry once
    if parsed.get('bio') == default_profile()['bio']:
        strict_messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " STRICT MODE: Return ONLY raw JSON matching schema. No commentary."},
            {"role": "user", "content": build_photo_messages(req.photo_urls, req.preferences)[-1]['content']}
        ]
        strict_content = chat_completion(strict_messages, diagnostics)
        if strict_content:
            strict_parsed = parse_response(strict_content)
            if strict_parsed.get('bio') != default_profile()['bio']:
                parsed = strict_parsed
    # Synthetic diversification as last resort
    if parsed.get('bio') == default_profile()['bio']:
        import random
        adjectives = ["Driven", "Adventurous", "Grounded", "Curious", "Balanced", "Empathetic", "Focused"]
        hobbies = ["Travel", "Fitness", "Cooking", "Outdoors", "Live Music", "Reading", "Tech"]
        random.shuffle(adjectives)
        random.shuffle(hobbies)
        parsed = {
            "bio": f"Blending {adjectives[0].lower()} energy with a love for {hobbies[0].lower()} and {hobbies[1].lower()}. Always up for growth, real conversation and spontaneous plans.",
            "traits": adjectives[:5],
            "interests": hobbies[:5]
        }
    # Approximate insights for scoring
    insights_like = {
        'avg_aesthetic_score': 0.6,
        'activities': [],
        'styles': [],
        'photo_count': len(req.photo_urls)
    }
    match_pct = assess_match_percentage(insights_like)
    strength = profile_strength(match_pct)
    resp = GenerateProfileResponse(
        bio=parsed['bio'],
        traits=parsed.get('traits', [])[:5],
        interests=parsed.get('interests', [])[:5],
        match_percentage=match_pct,
        profile_strength=strength,
        model_used=diagnostics.get('model_used', OPENAI_MODEL),
        openai_latency_ms=(diagnostics.get('attempts', [{}])[-1].get('latency_ms') if diagnostics.get('attempts') else None),
        diagnostics=diagnostics if DEBUG else None
    )
    return resp

# ---------------------------------------------------------------------------
# Simple root & health
# ---------------------------------------------------------------------------
@app.get('/')
async def root():
    return {"service": "flaire-profile-generator", "model": OPENAI_MODEL}

@app.get('/health')
async def health():
    return {"status": "ok", "model": OPENAI_MODEL}

# ---------------------------------------------------------------------------
# Run (optional)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('flaire_app:app', host='0.0.0.0', port=8100, reload=True)

# ---------------------------------------------------------------------------
# Diagnostics: list routes at startup and provide /debug/routes endpoint
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _log_routes():
    try:
        for r in app.router.routes:
            if hasattr(r, 'methods') and hasattr(r, 'path'):
                logger.info("[ROUTE] %s %s", ','.join(sorted(r.methods or [])), getattr(r, 'path', ''))
    except Exception as e:
        logger.warning("Failed to log routes: %s", e)

@app.get('/debug/routes')
async def debug_routes():
    return [
        {
            'path': getattr(r, 'path', ''),
            'methods': sorted(list(r.methods)) if hasattr(r, 'methods') and r.methods else [],
            'name': getattr(r, 'name', '')
        }
        for r in app.router.routes
        if getattr(r, 'path', '').startswith('/')
    ]
