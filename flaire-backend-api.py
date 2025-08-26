"""Legacy monolithic backend removed.

This placeholder file remains only to avoid broken references if any docs point here.
All functionality lives in flaire_app.py.
Safe to delete after verification.
"""

__all__: list[str] = []

# Initialize FastAPI app
app = FastAPI(
    title="Flaire Profile Builder API",
    description="AI-powered dating profile optimization backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",  # Vite dev server
        "https://flaire.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/flaire_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Stripe setup
stripe.api_key = "sk_test_your_stripe_secret_key"

# Logging (initialize early so we can log config warnings)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env values explicitly (do NOT fall back to exported environment variables for the OpenAI key)
load_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
if not os.path.exists(load_path):
    logger.error(f".env file not found at {load_path}. Create it and define OPENAI_API_KEY=<your_key>.")
    raise SystemExit(1)

env_values = dotenv_values(load_path)
openai_api_key = env_values.get("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY missing from .env file. Add it and restart.")
    raise SystemExit(1)

openai.api_key = openai_api_key

# OpenAI model configuration (read from .env first, fallback to default). We intentionally do NOT read from process env.
OPENAI_MODEL = env_values.get("OPENAI_MODEL", "gpt-3.5-turbo")
DEBUG_OPENAI_DIAGNOSTICS = env_values.get("OPENAI_DEBUG_DIAGNOSTICS", "0") in ("1", "true", "True", "YES", "yes")

masked_key = f"{openai_api_key[:4]}...{openai_api_key[-4:]}" if len(openai_api_key) > 8 else "(hidden)"
logger.info(f"Loaded OpenAI API key from .env (masked): {masked_key}")
logger.info(f"Using OpenAI model: {OPENAI_MODEL}")
logger.info(f"OpenAI version detected: {getattr(openai, '__version__', 'unknown')}")
if DEBUG_OPENAI_DIAGNOSTICS:
    logger.info("OpenAI detailed diagnostics ENABLED (OPENAI_DEBUG_DIAGNOSTICS=1)")
else:
    logger.info("OpenAI detailed diagnostics disabled. Set OPENAI_DEBUG_DIAGNOSTICS=1 in .env to enable.")

# AWS S3 setup
s3_client = boto3.client(
    's3',
    aws_access_key_id='your_aws_access_key',
    aws_secret_access_key='your_aws_secret_key',
    region_name='us-east-1'
)
S3_BUCKET = "flaire-user-photos"

# Celery setup for background tasks
celery = Celery('flaire', broker='redis://localhost:6379')

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# (logging already configured above)

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    plan = Column(String, default="free")  # free, pro, elite
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    profiles = relationship("Profile", back_populates="user")
    usage = relationship("Usage", back_populates="user")

class Profile(Base):
    __tablename__ = "profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    bio = Column(Text)
    traits = Column(Text)  # JSON string
    interests = Column(Text)  # JSON string
    match_percentage = Column(Integer)
    profile_strength = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="profiles")
    photos = relationship("Photo", back_populates="profile")

class Photo(Base):
    __tablename__ = "photos"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    filename = Column(String)
    s3_url = Column(String)
    analysis = Column(Text)  # JSON string with photo analysis
    created_at = Column(DateTime, default=datetime.utcnow)
    
    profile = relationship("Profile", back_populates="photos")

class Usage(Base):
    __tablename__ = "usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action_type = Column(String)  # profile_generation, opener_generation
    count = Column(Integer, default=0)
    date = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="usage")

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    plan: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class ProfileAnalysisRequest(BaseModel):
    photo_urls: List[str]

class ProfileResponse(BaseModel):
    id: str
    bio: str
    traits: List[str]
    interests: List[str]
    match_percentage: int
    profile_strength: str
    created_at: datetime

class PhotoAnalysis(BaseModel):
    filename: str
    analysis: Dict[str, Any]

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Photo analysis functions
class PhotoAnalyzer:
    def __init__(self):
        # Initialize AI models for photo analysis
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Emotion model (optional) - using custom model id placeholder which may not exist; wrap safe
        try:
            self.emotion_pipeline = pipeline("image-classification", model="fer2013_mini_XCEPTION.102-0.66.hdf5")
        except Exception as e:
            logger.warning(f"Emotion model not loaded: {e}. Proceeding without emotion analysis.")
            self.emotion_pipeline = None
        
        # BLIP model for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    async def analyze_photo(self, image_path: str) -> Dict[str, Any]:
        """Analyze a photo for various attributes"""
        try:
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            analysis = {}
            
            # Face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            analysis['faces_count'] = len(faces)
            analysis['has_face'] = len(faces) > 0
            
            # Image quality metrics
            analysis['image_quality'] = self._assess_image_quality(image)
            
            # Scene analysis using BLIP
            inputs = self.blip_processor(pil_image, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_new_tokens=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            analysis['scene_description'] = caption
            
            # Activity/hobby detection
            analysis['detected_activities'] = self._detect_activities(caption)
            
            # Aesthetic score
            analysis['aesthetic_score'] = self._calculate_aesthetic_score(image)
            
            # Outfit/style analysis
            analysis['style_analysis'] = self._analyze_style(caption, image_rgb)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing photo: {str(e)}")
            return {"error": str(e)}
    
    def _assess_image_quality(self, image) -> Dict[str, float]:
        """Assess technical quality of image"""
        # Blur detection using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness assessment
        brightness = np.mean(image)
        
        # Contrast assessment
        contrast = image.std()
        
        return {
            'sharpness': min(blur_score / 1000, 1.0),  # Normalize to 0-1
            'brightness': brightness / 255,
            'contrast': contrast / 128
        }
    
    def _detect_activities(self, caption: str) -> List[str]:
        """Extract activities/hobbies from image caption"""
        activity_keywords = {
            'hiking': ['mountain', 'trail', 'hiking', 'nature', 'outdoor'],
            'travel': ['city', 'building', 'street', 'tourist', 'vacation'],
            'fitness': ['gym', 'exercise', 'workout', 'sports', 'running'],
            'cooking': ['kitchen', 'food', 'cooking', 'restaurant', 'meal'],
            'music': ['guitar', 'piano', 'concert', 'music', 'instrument'],
            'art': ['painting', 'drawing', 'art', 'gallery', 'creative'],
            'pets': ['dog', 'cat', 'pet', 'animal'],
            'beach': ['beach', 'ocean', 'water', 'swimming', 'surf']
        }
        
        detected = []
        caption_lower = caption.lower()
        
        for activity, keywords in activity_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                detected.append(activity)
        
        return detected
    
    def _calculate_aesthetic_score(self, image) -> float:
        """Calculate aesthetic appeal score"""
        # Rule of thirds
        height, width = image.shape[:2]
        thirds_h = height // 3
        thirds_w = width // 3
        
        # Simple aesthetic heuristics
        score = 0.5  # Base score
        
        # Color diversity
        unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
        color_diversity = min(unique_colors / 10000, 1.0)
        score += color_diversity * 0.3
        
        # Edge density (interesting composition)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        score += min(edge_density * 2, 0.2)
        
        return min(score, 1.0)
    
    def _analyze_style(self, caption: str, image) -> Dict[str, Any]:
        """Analyze clothing style and overall appearance"""
        style_indicators = {
            'casual': ['casual', 'jeans', 't-shirt', 'sneakers'],
            'professional': ['suit', 'tie', 'business', 'formal'],
            'sporty': ['sports', 'athletic', 'gym', 'workout'],
            'trendy': ['stylish', 'fashionable', 'modern'],
            'outdoorsy': ['outdoor', 'hiking', 'nature', 'adventure']
        }
        
        caption_lower = caption.lower()
        detected_styles = []
        
        for style, indicators in style_indicators.items():
            if any(indicator in caption_lower for indicator in indicators):
                detected_styles.append(style)
        
        return {
            'styles': detected_styles,
            'dominant_colors': self._get_dominant_colors(image)
        }
    
    def _get_dominant_colors(self, image) -> List[str]:
        """Extract dominant colors from image"""
        # Reshape image to be a list of pixels
        pixels = image.reshape((-1, 3))
        
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        colors = []
        for center in kmeans.cluster_centers_:
            # Convert BGR to color name (simplified)
            b, g, r = center
            if r > 150 and g < 100 and b < 100:
                colors.append('red')
            elif g > 150 and r < 100 and b < 100:
                colors.append('green')
            elif b > 150 and r < 100 and g < 100:
                colors.append('blue')
            elif r > 150 and g > 150 and b < 100:
                colors.append('yellow')
            elif r > 100 and g > 100 and b > 100:
                colors.append('neutral')
            else:
                colors.append('dark')
        
        return colors[:2]  # Return top 2 dominant colors

# Profile generation using OpenAI
class ProfileGenerator:
    def __init__(self):
        # We keep no persistent client to avoid version-specific constructor issues
        # (e.g., unexpected 'proxies' kw). We'll dynamically choose an API path.
        self.last_model_used: Optional[str] = None  # track which model ultimately produced content
        self.last_diagnostics: Optional[Dict[str, Any]] = None
        # Disable new OpenAI SDK client path by default (proxies constructor issue). Enable with OPENAI_ENABLE_SDK=1
        self._new_client_disabled = os.getenv('OPENAI_ENABLE_SDK', '0') not in ('1','true','TRUE','yes','YES')
        # Additional initialization can go here

    def _supports_legacy_chat(self):
        # Restrict legacy usage strictly to <1.x to avoid deprecation spam
        version_str = getattr(openai, '__version__', '1.0.0')
        try:
            major = int(version_str.split('.')[0])
        except Exception:
            major = 1
        return major < 1 and hasattr(openai, 'ChatCompletion') and hasattr(openai.ChatCompletion, 'create')

    async def _call_legacy_chat(self, prompt: str) -> Optional[str]:
        # Only attempt if pre-1.x library actually present
        if not self._supports_legacy_chat():
            return None
        loop = asyncio.get_event_loop()
        def _do():
            try:
                return openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a dating profile expert who creates attractive, authentic profiles for men that get matches."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_completion_tokens=800
                ).choices[0].message.content
            except Exception as e:
                logger.warning(f"Legacy ChatCompletion failed (pre-1.x only): {e}")
                return None
        return await loop.run_in_executor(None, _do)

    async def _call_new_client(self, prompt: str) -> Optional[str]:
        """Attempt OpenAI 1.x style calls with model fallback list and pre-validation."""
        if self._new_client_disabled:
            logger.debug("[OPENAI_DIAG] Skipping new client path (previous fatal error).")
            return None
        self.last_diagnostics = {
            'phase': 'init',
            'primary_model': OPENAI_MODEL,
            'openai_version': getattr(openai, '__version__', 'unknown'),
            'attempts': [],
            'import_error': None,
            'interface_missing': False,
            'models_filtered': None,
            'all_failed': False
        }
        try:
            from openai import OpenAI
            client = OpenAI()  # May raise due to unexpected kwargs injected by patched libs
        except Exception as e:
            self.last_diagnostics['import_error'] = str(e)
            logger.warning(f"[OPENAI_DIAG] New client import/init failed: {e} -> enabling legacy fallback path")
            # If error signature suggests incompatible constructor (e.g. proxies arg), permanently disable new path this process
            if 'unexpected keyword argument' in str(e) and 'proxies' in str(e):
                self._new_client_disabled = True
            return None

        if not (hasattr(client, 'chat') and hasattr(client.chat, 'completions')):
            self.last_diagnostics['interface_missing'] = True
            logger.warning("[OPENAI_DIAG] OpenAI client chat.completions interface not present")
            return None

        primary = OPENAI_MODEL
        # Order fallbacks with most broadly accessible / legacy-compatible models first
        fallback_candidates = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
        ]
        models_to_try = []
        seen = set()
        for m in [primary] + fallback_candidates:
            if m and m not in seen:
                seen.add(m)
                models_to_try.append(m)

        # Optional: attempt to list models to prune obviously invalid ones (ignore errors)
        try:
            loop = asyncio.get_event_loop()
            def _list_models():
                try:
                    return {mdl.id for mdl in client.models.list().data}
                except Exception:
                    return None
            available = await loop.run_in_executor(None, _list_models)
            if available:
                before_len = len(models_to_try)
                models_to_try = [m for m in models_to_try if m in available]
                if not models_to_try:
                    models_to_try = [primary] + [m for m in fallback_candidates if m != primary]
                    removed = before_len  # everything removed
                else:
                    removed = before_len - len(models_to_try)
                    if removed:
                        logger.info(f"Filtered out {removed} unavailable model names via models.list().")
                self.last_diagnostics['models_filtered'] = {'available_count': len(available), 'removed': removed}
        except Exception as e:
            logger.debug(f"Model listing skipped due to error: {e}")

        last_error: Optional[Exception] = None
        loop = asyncio.get_event_loop()
        attempt_counter = 0
        def _attempt(model_name: str):
            nonlocal last_error
            try:
                start_t = time.perf_counter()
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a dating profile expert who creates attractive, authentic profiles for men that get matches."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_completion_tokens=800
                )
                latency = (time.perf_counter() - start_t) * 1000
                return model_name, resp.choices[0].message.content, latency
            except Exception as e:
                last_error = e
                return None
        correlation_id = uuid.uuid4().hex[:12]
        for m in models_to_try:
            attempt_counter += 1
            logger.info(f"[OID:{correlation_id}] Attempt {attempt_counter}/{len(models_to_try)} model='{m}'")
            start_wall = time.perf_counter()
            result = await loop.run_in_executor(None, _attempt, m)
            wall_latency = (time.perf_counter() - start_wall) * 1000
            if result:
                used_model, content, api_latency = result
                logger.info(
                    f"[OID:{correlation_id}] SUCCESS model='{used_model}' api_latency_ms={api_latency:.1f} wall_ms={wall_latency:.1f}"
                )
                if used_model != primary:
                    logger.info(f"[OID:{correlation_id}] Fallback success. primary='{primary}' used='{used_model}'")
                self.last_model_used = used_model
                self.last_diagnostics['attempts'].append({
                    'model': used_model,
                    'success': True,
                    'api_latency_ms': round(api_latency, 2),
                    'wall_latency_ms': round(wall_latency, 2)
                })
                self.last_diagnostics['phase'] = 'success'
                return content
            else:
                if last_error:
                    err_type = type(last_error).__name__
                    logger.warning(
                        f"[OID:{correlation_id}] FAIL model='{m}' wall_ms={wall_latency:.1f} error_type={err_type} error={last_error}"
                    )
                    if DEBUG_OPENAI_DIAGNOSTICS:
                        import traceback
                        tb = ''.join(traceback.format_exception(type(last_error), last_error, last_error.__traceback__))
                        logger.debug(f"[OID:{correlation_id}] TRACE model='{m}'\n{tb}")
                    self.last_diagnostics['attempts'].append({
                        'model': m,
                        'success': False,
                        'wall_latency_ms': round(wall_latency, 2),
                        'error_type': err_type,
                        'error_message': str(last_error)[:300]
                    })
        if last_error:
            logger.warning(f"[OID:{correlation_id}] All model attempts exhausted. last_error={type(last_error).__name__}: {last_error}")
            self.last_diagnostics['all_failed'] = True
            self.last_diagnostics['phase'] = 'failed'
        return None

    async def _call_http_fallback(self, prompt: str) -> Optional[str]:
        """Minimal HTTP fallback for chat completion (avoids SDK constructor issues)."""
        if not openai.api_key:
            return None
        url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions')
        headers = {
            'Authorization': f'Bearer {openai.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': OPENAI_MODEL,
            'messages': [
                {"role": "system", "content": "You are a dating profile expert who creates attractive, authentic profiles for men that get matches."},
                {"role": "user", "content": prompt}
            ],
            'temperature': 0.7,
            'max_tokens': 800
        }
        start = time.perf_counter()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            if self.last_diagnostics is not None:
                self.last_diagnostics['http_fallback'] = {
                    'status_code': resp.status_code,
                    'latency_ms': round(elapsed, 2)
                }
            if resp.status_code != 200:
                logger.warning(f"[OPENAI_HTTP] status={resp.status_code} body={resp.text[:160]}")
                if self.last_diagnostics is not None:
                    self.last_diagnostics['http_fallback']['error'] = resp.text[:500]
                return None
            data = resp.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content')
            if self.last_diagnostics is not None:
                self.last_diagnostics['http_fallback']['success'] = bool(content)
            return content
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(f"[OPENAI_HTTP] request failed after {elapsed:.1f}ms: {e}")
            if self.last_diagnostics is not None:
                self.last_diagnostics['http_fallback'] = {
                    'exception': str(e),
                    'latency_ms': round(elapsed, 2)
                }
            return None

    async def generate_profile(self, photo_analyses: List[Dict], user_preferences: Dict = None) -> Dict[str, Any]:
        """Generate dating profile using OpenAI with resilient multi-path strategy."""
        try:
            insights = self._compile_insights(photo_analyses)
            prompt = self._create_generation_prompt(insights, user_preferences)

            if not openai.api_key:
                logger.info("No OPENAI_API_KEY; returning default profile.")
                profile_data = self._get_default_profile()
            else:
                self.last_model_used = None
                diag_start = time.perf_counter()
                content = await self._call_new_client(prompt)
                openai_latency_ms = (time.perf_counter() - diag_start) * 1000
                # Only attempt legacy path if using pre-1.x library
                if content is None and self._supports_legacy_chat():
                    content = await self._call_legacy_chat(prompt)
                    if content:
                        self.last_model_used = self.last_model_used or OPENAI_MODEL  # approximate
                # HTTP fallback (direct REST) if SDK path skipped or failed
                if content is None:
                    if not hasattr(self, '_call_http_fallback'):
                        # Define lightweight HTTP fallback dynamically if missing
                        async def _http_fb(prompt_inner: str):
                            try:
                                import requests, json as _json, time as _time
                                url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1/chat/completions')
                                headers = {'Authorization': f'Bearer {openai.api_key}', 'Content-Type': 'application/json'}
                                payload = {
                                    'model': OPENAI_MODEL,
                                    'messages': [
                                        {"role": "system", "content": "You are a dating profile expert who creates attractive, authentic profiles for men that get matches."},
                                        {"role": "user", "content": prompt_inner}
                                    ],
                                    'temperature': 0.7,
                                    'max_tokens': 800
                                }
                                start_req = _time.perf_counter()
                                r = requests.post(url, headers=headers, json=payload, timeout=30)
                                if r.status_code == 200:
                                    data = r.json()
                                    return data.get('choices', [{}])[0].get('message', {}).get('content')
                            except Exception as _e:
                                logger.debug(f"[OPENAI_HTTP] fallback exception: {_e}")
                            return None
                        self._call_http_fallback = _http_fb  # type: ignore
                    content = await self._call_http_fallback(prompt)  # type: ignore
                    if content:
                        self.last_model_used = self.last_model_used or OPENAI_MODEL
                if content is None:
                    logger.warning(
                        f"All OpenAI calls failed; using default profile fallback. (Set a valid OPENAI_API_KEY and OPENAI_MODEL) latency_ms={openai_latency_ms:.1f} model_chain_last={self.last_model_used}"
                    )
                    content = json.dumps(self._get_default_profile())
                profile_data = self._parse_profile_response(content)

            profile_data['match_percentage'] = self._calculate_match_percentage(insights)
            profile_data['profile_strength'] = self._assess_profile_strength(profile_data)
            if self.last_model_used:
                profile_data['model_used'] = self.last_model_used
            else:
                profile_data['model_used'] = 'fallback_default'
            if 'openai_latency_ms' not in profile_data and 'openai_latency_ms' in locals():
                profile_data['openai_latency_ms'] = round(openai_latency_ms, 2)
            if DEBUG_OPENAI_DIAGNOSTICS and self.last_diagnostics:
                # Attach a trimmed diagnostics view
                diag_copy = dict(self.last_diagnostics)
                # Potentially large; trim attempts if huge
                if len(diag_copy.get('attempts', [])) > 10:
                    diag_copy['attempts'] = diag_copy['attempts'][:10]
                    diag_copy['truncated'] = True
                profile_data['openai_diagnostics'] = diag_copy
            try:
                logger.info(
                    "Generated profile (authenticated=%s, model=%s, openai_latency_ms=%.1f): %s",
                    bool(user_preferences),
                    profile_data.get('model_used'),
                    profile_data.get('openai_latency_ms', openai_latency_ms if 'openai_latency_ms' in locals() else -1),
                    json.dumps(profile_data, ensure_ascii=False)
                )
            except Exception:
                pass
            return profile_data
        except Exception as e:
            logger.error(f"Error generating profile: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to generate profile")
    
    def _compile_insights(self, photo_analyses: List[Dict]) -> Dict[str, Any]:
        """Compile insights from multiple photo analyses"""
        all_activities = []
        all_styles = []
        avg_aesthetic = 0
        total_faces = 0
        
        for analysis in photo_analyses:
            if 'detected_activities' in analysis:
                all_activities.extend(analysis['detected_activities'])
            if 'style_analysis' in analysis and 'styles' in analysis['style_analysis']:
                all_styles.extend(analysis['style_analysis']['styles'])
            if 'aesthetic_score' in analysis:
                avg_aesthetic += analysis['aesthetic_score']
            if 'faces_count' in analysis:
                total_faces += analysis['faces_count']
        
        return {
            'activities': list(set(all_activities)),
            'styles': list(set(all_styles)),
            'avg_aesthetic_score': avg_aesthetic / len(photo_analyses) if photo_analyses else 0,
            'total_faces': total_faces,
            'photo_count': len(photo_analyses)
        }
    
    def _create_generation_prompt(self, insights: Dict, preferences: Dict = None) -> str:
        """Create prompt for profile generation"""
        activities = ', '.join(insights.get('activities', []))
        styles = ', '.join(insights.get('styles', []))
        
        prompt = f"""
        Based on the following analysis of dating profile photos, create an attractive, authentic dating profile:

        Detected Activities/Hobbies: {activities}
        Style Analysis: {styles}
        Aesthetic Score: {insights.get('avg_aesthetic_score', 0):.2f}
        Number of Photos: {insights.get('photo_count', 0)}

        Create a dating profile with:
        1. A compelling bio (150-300 characters) that's witty, confident, and authentic
        2. 5 personality traits that match the photo analysis
        3. 5 interests/hobbies based on the detected activities

        Format as JSON:
        {{
            "bio": "...",
            "traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
            "interests": ["interest1", "interest2", "interest3", "interest4", "interest5"]
        }}

        Make it masculine, confident, and results-oriented. Focus on what makes this person attractive and interesting.
        """
        
        return prompt
    
    def _parse_profile_response(self, response: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured data"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._fallback_parse(response)
        except Exception as e:
            logger.error(f"Error parsing profile response: {str(e)}")
            return self._get_default_profile()

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Heuristic fallback parsing when JSON isn't clean."""
        bio = ""
        traits: List[str] = []
        interests: List[str] = []
        try:
            lines = [l.strip('- *') for l in response.splitlines() if l.strip()]
            for l in lines:
                low = l.lower()
                if not bio and len(l) > 40:
                    bio = l[:300]
                if 'trait' in low and ':' in l:
                    parts = l.split(':', 1)[1]
                    traits.extend([p.strip().title() for p in re.split(r',|;|\|', parts) if p.strip()])
                if 'interest' in low and ':' in l:
                    parts = l.split(':', 1)[1]
                    interests.extend([p.strip().title() for p in re.split(r',|;|\|', parts) if p.strip()])
        except Exception:
            pass
        default = self._get_default_profile()
        return {
            'bio': bio or default['bio'],
            'traits': traits[:5] or default['traits'],
            'interests': interests[:5] or default['interests']
        }
    
    def _calculate_match_percentage(self, insights: Dict) -> int:
        """Calculate estimated match percentage"""
        score = 70  # Base score
        
        # Photo quality bonus
        if insights.get('avg_aesthetic_score', 0) > 0.7:
            score += 15
        elif insights.get('avg_aesthetic_score', 0) > 0.5:
            score += 10
        
        # Activity diversity bonus
        activity_count = len(insights.get('activities', []))
        score += min(activity_count * 2, 10)
        
        # Photo count bonus
        photo_count = insights.get('photo_count', 0)
        if photo_count >= 4:
            score += 5
        
        return min(score, 95)
    
    def _assess_profile_strength(self, profile_data: Dict) -> str:
        """Assess overall profile strength"""
        match_pct = profile_data.get('match_percentage', 0)
        
        if match_pct >= 90:
            return "Outstanding"
        elif match_pct >= 80:
            return "Excellent"
        elif match_pct >= 70:
            return "Very Good"
        elif match_pct >= 60:
            return "Good"
        else:
            return "Needs Improvement"
    
    def _get_default_profile(self) -> Dict[str, Any]:
        """Return default profile if generation fails"""
        return {
            "bio": "Adventure seeker with a passion for life. Always up for trying something new and making genuine connections. Let's grab coffee and see where the conversation takes us! ☕",
            "traits": ["Adventurous", "Authentic", "Confident", "Social", "Ambitious"],
            "interests": ["Travel", "Coffee", "Fitness", "Food", "Adventure"]
        }

# Initialize analyzers
photo_analyzer = PhotoAnalyzer()
profile_generator = ProfileGenerator()

# API Routes

@app.post("/auth/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=str(db_user.id),
        email=db_user.email,
        full_name=db_user.full_name,
        plan=db_user.plan,
        created_at=db_user.created_at
    )

@app.post("/auth/login", response_model=Token)
async def login_user(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Login user and return access token"""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/photos/upload")
async def upload_photos(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and analyze photos"""
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 photos allowed")
    
    uploaded_photos = []
    
    for file in files:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
        
        # Generate unique filename
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Save temporarily for processing
        temp_path = f"/tmp/{unique_filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Upload to S3
        s3_key = f"user-photos/{current_user.id}/{unique_filename}"
        s3_client.upload_file(temp_path, S3_BUCKET, s3_key)
        s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        
        # Analyze photo
        analysis = await photo_analyzer.analyze_photo(temp_path)
        
        # Save to database
        db_photo = Photo(
            user_id=current_user.id,
            filename=unique_filename,
            s3_url=s3_url,
            analysis=json.dumps(analysis)
        )
        db.add(db_photo)
        
        uploaded_photos.append({
            "id": str(db_photo.id),
            "filename": unique_filename,
            "s3_url": s3_url,
            "analysis": analysis
        })
        
        # Clean up temp file
        os.remove(temp_path)
    
    db.commit()
    
    return {"photos": uploaded_photos}

@app.post("/photos/upload-public")
async def upload_photos_public(
    files: List[UploadFile] = File(...)
):
    """Upload and analyze photos without authentication (no persistence).
    Returns immediate analysis results for trial/unauthenticated users.
    """
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 photos allowed")
    results = []
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
        file_extension = file.filename.split('.')[-1]
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        temp_path = f"/tmp/{unique_filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        # Analyze locally
        analysis = await photo_analyzer.analyze_photo(temp_path)
        try:
            os.remove(temp_path)
        except Exception:
            pass
        results.append({
            "id": str(uuid.uuid4()),  # ephemeral id (not stored)
            "filename": unique_filename,
            "analysis": analysis
        })
    return {"photos": results, "persisted": False}

@app.post("/profile/generate", response_model=ProfileResponse)
async def generate_profile(
    photo_ids: List[str] = Body(..., embed=False, description="List of photo UUIDs"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate dating profile from uploaded photos"""
    # Check usage limits
    usage_limit = await check_usage_limit(current_user, "profile_generation", db)
    if not usage_limit:
        raise HTTPException(status_code=402, detail="Usage limit exceeded. Please upgrade your plan.")
    
    # Get photo analyses
    photos = db.query(Photo).filter(
        Photo.id.in_(photo_ids),
        Photo.user_id == current_user.id
    ).all()
    
    if not photos:
        raise HTTPException(status_code=400, detail="No valid photos found")
    
    # Parse photo analyses
    photo_analyses = []
    for photo in photos:
        try:
            analysis = json.loads(photo.analysis)
            photo_analyses.append(analysis)
        except:
            continue
    
    # Generate profile
    profile_data = await profile_generator.generate_profile(photo_analyses)
    
    # Save profile to database
    db_profile = Profile(
        user_id=current_user.id,
        bio=profile_data['bio'],
        traits=json.dumps(profile_data['traits']),
        interests=json.dumps(profile_data['interests']),
        match_percentage=profile_data['match_percentage'],
        profile_strength=profile_data['profile_strength']
    )
    db.add(db_profile)
    
    # Link photos to profile
    for photo in photos:
        photo.profile_id = db_profile.id
    
    # Update usage
    await update_usage(current_user.id, "profile_generation", db)
    
    db.commit()
    db.refresh(db_profile)
    
    return ProfileResponse(
        id=str(db_profile.id),
        bio=db_profile.bio,
        traits=json.loads(db_profile.traits),
        interests=json.loads(db_profile.interests),
        match_percentage=db_profile.match_percentage,
        profile_strength=db_profile.profile_strength,
        created_at=db_profile.created_at
    )

@app.post("/profile/generate-public")
async def generate_profile_public(
    analyses: List[Dict[str, Any]] = Body(..., description="List of photo analysis dicts from /photos/upload-public"),
):
    """Generate a preview dating profile without authentication using provided photo analyses.
    Does not persist any data and uses the configured OpenAI model (chatgpt-5 by default).
    """
    if not analyses:
        raise HTTPException(status_code=400, detail="No analyses provided")
    try:
        profile_data = await profile_generator.generate_profile(analyses)
        # Mark as preview
        profile_data['profile_strength'] = f"{profile_data.get('profile_strength','Preview')} (Preview)"
        try:
            logger.info("Generated public preview profile: %s", json.dumps(profile_data, ensure_ascii=False))
        except Exception:
            pass
        return profile_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Public profile generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate preview profile")

@app.get("/profile/history")
async def get_profile_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's profile generation history"""
    profiles = db.query(Profile).filter(Profile.user_id == current_user.id).order_by(Profile.created_at.desc()).all()
    
    return {
        "profiles": [
            {
                "id": str(profile.id),
                "bio": profile.bio,
                "traits": json.loads(profile.traits),
                "interests": json.loads(profile.interests),
                "match_percentage": profile.match_percentage,
                "profile_strength": profile.profile_strength,
                "created_at": profile.created_at
            }
            for profile in profiles
        ]
    }

@app.get("/user/usage")
async def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's current usage statistics"""
    today = datetime.utcnow().date()
    
    profile_usage = db.query(Usage).filter(
        Usage.user_id == current_user.id,
        Usage.action_type == "profile_generation",
        Usage.date >= today
    ).first()
    
    opener_usage = db.query(Usage).filter(
        Usage.user_id == current_user.id,
        Usage.action_type == "opener_generation",
        Usage.date >= today
    ).first()
    
    limits = {
        "free": {"profiles": 2, "openers": 5},
        "pro": {"profiles": float('inf'), "openers": 100},
        "elite": {"profiles": float('inf'), "openers": float('inf')}
    }
    
    user_limits = limits.get(current_user.plan, limits["free"])
    
    return {
        "plan": current_user.plan,
        "usage": {
            "profiles": profile_usage.count if profile_usage else 0,
            "openers": opener_usage.count if opener_usage else 0
        },
        "limits": user_limits,
        "remaining": {
            "profiles": max(0, user_limits["profiles"] - (profile_usage.count if profile_usage else 0)) if user_limits["profiles"] != float('inf') else float('inf'),
            "openers": max(0, user_limits["openers"] - (opener_usage.count if opener_usage else 0)) if user_limits["openers"] != float('inf') else float('inf')
        }
    }

# Helper functions for usage management
async def check_usage_limit(user: User, action_type: str, db: Session) -> bool:
    """Check if user has exceeded usage limits"""
    limits = {
        "free": {"profile_generation": 2, "opener_generation": 5},
        "pro": {"profile_generation": float('inf'), "opener_generation": 100},
        "elite": {"profile_generation": float('inf'), "opener_generation": float('inf')}
    }
    
    user_limit = limits.get(user.plan, limits["free"]).get(action_type, 0)
    
    if user_limit == float('inf'):
        return True
    
    today = datetime.utcnow().date()
    usage = db.query(Usage).filter(
        Usage.user_id == user.id,
        Usage.action_type == action_type,
        Usage.date >= today
    ).first()
    
    current_usage = usage.count if usage else 0
    return current_usage < user_limit

async def update_usage(user_id: str, action_type: str, db: Session):
    """Update user's usage count"""
    today = datetime.utcnow().date()
    usage = db.query(Usage).filter(
        Usage.user_id == user_id,
        Usage.action_type == action_type,
        Usage.date >= today
    ).first()
    
    if usage:
        usage.count += 1
    else:
        usage = Usage(
            user_id=user_id,
            action_type=action_type,
            count=1
        )
        db.add(usage)

# Stripe webhook handling
@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, "whsec_your_webhook_secret"
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle subscription events
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        await handle_successful_payment(session, db)
    elif event['type'] == 'customer.subscription.updated':
        subscription = event['data']['object']
        await handle_subscription_update(subscription, db)
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        await handle_subscription_cancellation(subscription, db)
    
    return {"status": "success"}

async def handle_successful_payment(session, db: Session):
    """Handle successful payment completion"""
    customer_email = session.get('customer_email')
    mode = session.get('mode')
    
    if mode == 'subscription':
        # Update user's plan
        user = db.query(User).filter(User.email == customer_email).first()
        if user:
            # Determine plan based on price_id or amount
            amount = session.get('amount_total', 0) / 100  # Convert from cents
            if amount >= 24.99:
                user.plan = 'elite'
            elif amount >= 14.99:
                user.plan = 'pro'
            
            db.commit()
            logger.info(f"Updated user {user.email} to {user.plan} plan")

async def handle_subscription_update(subscription, db: Session):
    """Handle subscription plan changes"""
    customer_id = subscription.get('customer')
    status = subscription.get('status')
    
    # Get user by Stripe customer ID (you'd need to store this in User model)
    if status == 'active':
        # Update plan based on subscription items
        pass

async def handle_subscription_cancellation(subscription, db: Session):
    """Handle subscription cancellation"""
    customer_id = subscription.get('customer')
    
    # Downgrade user to free plan
    # Implementation depends on how you store Stripe customer IDs

# Background tasks with Celery
@celery.task
def process_photo_analysis_async(photo_id: str, user_id: str, file_path: str):
    """Process photo analysis asynchronously"""
    try:
        # This would run the photo analysis in background
        # and update the database when complete
        analysis = photo_analyzer.analyze_photo(file_path)
        
        # Update database with analysis results
        db = SessionLocal()
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if photo:
            photo.analysis = json.dumps(analysis)
            db.commit()
        db.close()
        
        return {"status": "success", "photo_id": photo_id}
        
    except Exception as e:
        logger.error(f"Background photo analysis failed: {str(e)}")
        return {"status": "error", "error": str(e)}

@celery.task
def generate_profile_async(user_id: str, photo_ids: List[str]):
    """Generate profile asynchronously"""
    try:
        db = SessionLocal()
        
        # Get photo analyses
        photos = db.query(Photo).filter(
            Photo.id.in_(photo_ids),
            Photo.user_id == user_id
        ).all()
        
        photo_analyses = []
        for photo in photos:
            try:
                analysis = json.loads(photo.analysis)
                photo_analyses.append(analysis)
            except:
                continue
        
        # Generate profile
        profile_data = profile_generator.generate_profile(photo_analyses)
        
        # Save to database
        db_profile = Profile(
            user_id=user_id,
            bio=profile_data['bio'],
            traits=json.dumps(profile_data['traits']),
            interests=json.dumps(profile_data['interests']),
            match_percentage=profile_data['match_percentage'],
            profile_strength=profile_data['profile_strength']
        )
        db.add(db_profile)
        db.commit()
        
        db.close()
        return {"status": "success", "profile_id": str(db_profile.id)}
        
    except Exception as e:
        logger.error(f"Background profile generation failed: {str(e)}")
        return {"status": "error", "error": str(e)}

# Health check and monitoring endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Get application metrics"""
    # Total users
    total_users = db.query(User).count()
    
    # Active users (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_users = db.query(Usage).filter(Usage.date >= thirty_days_ago).distinct(Usage.user_id).count()
    
    # Profiles generated today
    today = datetime.utcnow().date()
    profiles_today = db.query(Profile).filter(Profile.created_at >= today).count()
    
    # Plan distribution
    plan_distribution = {}
    for plan in ['free', 'pro', 'elite']:
        plan_distribution[plan] = db.query(User).filter(User.plan == plan).count()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "profiles_generated_today": profiles_today,
        "plan_distribution": plan_distribution,
        "timestamp": datetime.utcnow()
    }

# Additional utility endpoints
@app.delete("/photos/{photo_id}")
async def delete_photo(
    photo_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a user's photo"""
    photo = db.query(Photo).filter(
        Photo.id == photo_id,
        Photo.user_id == current_user.id
    ).first()
    
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # Delete from S3
    try:
        s3_key = photo.s3_url.split(f"https://{S3_BUCKET}.s3.amazonaws.com/")[1]
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        logger.error(f"Failed to delete photo from S3: {str(e)}")
    
    # Delete from database
    db.delete(photo)
    db.commit()
    
    return {"message": "Photo deleted successfully"}

@app.put("/user/preferences")
async def update_user_preferences(
    preferences: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user preferences for profile generation"""
    # Store preferences in Redis for quick access
    redis_key = f"user_preferences:{current_user.id}"
    redis_client.set(redis_key, json.dumps(preferences), ex=86400 * 30)  # 30 days
    
    return {"message": "Preferences updated successfully"}

@app.get("/user/preferences")
async def get_user_preferences(current_user: User = Depends(get_current_user)):
    """Get user preferences"""
    redis_key = f"user_preferences:{current_user.id}"
    preferences = redis_client.get(redis_key)
    
    if preferences:
        return {"preferences": json.loads(preferences)}
    else:
        return {"preferences": {}}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "status_code": 500}
    )

# Database initialization
def create_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    # Create database tables
    create_tables()
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Example usage and testing
"""
# Requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
stripe==7.8.0
celery==5.3.4
boto3==1.34.0
Pillow==10.1.0
opencv-python==4.8.1.78
transformers==4.35.2
torch==2.1.1
torchvision==0.16.1
scikit-learn==1.3.2
aiofiles==23.2.1
openai==1.3.7

# Docker setup
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Environment variables (.env)
DATABASE_URL=postgresql://user:password@localhost/flaire_db
SECRET_KEY=your_secret_key_here
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET=flaire-user-photos
REDIS_URL=redis://localhost:6379
"""