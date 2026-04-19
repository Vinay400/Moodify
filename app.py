"""
Moodify — Emotion-Based Music Recommendation System
====================================================
Uses a CNN (FER-2013 architecture) + Haar Cascade for real-time facial
emotion recognition, then maps detected emotions to song recommendations
from the MUSE v3 dataset via Russell's circumplex model (valence × arousal).

Authors : Vinay
Stack   : Streamlit · TensorFlow/Keras · OpenCV · Pandas · Plotly
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter
from pathlib import Path
from urllib.parse import quote_plus
from keras.models import load_model

# ── Page Config (must be the very first Streamlit command) ───────────────────
st.set_page_config(
    page_title="Moodify | Emotion-Based Music Recommendation",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialisation ─────────────────────────────────────────────
_defaults = {
    "favorites": [],
    "mood_history": [],
    "last_predictions": None,   # aggregate prediction array from last scan
    "last_emotions": [],         # list of detected emotion labels
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Load Dataset ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset():
    """Load and prepare the MUSE v3 dataset."""
    raw = pd.read_csv(BASE_DIR / "muse_v3.csv")
    raw["name"] = raw["track"]
    raw["emotional"] = raw["number_of_emotion_tags"]
    raw["pleasant"] = raw["valence_tags"]
    raw["arousal"] = raw["arousal_tags"]
    raw["dominance"] = raw["dominance_tags"]
    # Build web links
    raw["web_link"] = raw.apply(
        lambda r: _to_web_link(r["lastfm_url"], r["name"], r["artist"]), axis=1
    )
    cols = ["name", "artist", "genre", "emotional", "pleasant", "arousal", "dominance", "web_link"]
    return raw[cols]


def _to_web_link(url, name, artist):
    """Convert a raw URL to a usable Spotify / Last.fm link."""
    try:
        if isinstance(url, str):
            if "api.spotify.com/v1/tracks/" in url:
                tid = url.rstrip("/").split("/")[-1].split("?")[0]
                return f"https://open.spotify.com/track/{tid}"
            if "open.spotify.com/track/" in url:
                return url
            if "lastfm" in url or "last.fm" in url:
                return url
        query = quote_plus(f"{name} {artist}")
        return f"https://open.spotify.com/search/{query}"
    except Exception:
        query = quote_plus(f"{name} {artist}")
        return f"https://open.spotify.com/search/{query}"


def to_youtube_search(name, artist):
    """Build a YouTube search URL for a given track."""
    return f"https://www.youtube.com/results?search_query={quote_plus(f'{name} {artist}')}"


df = load_dataset()

# Top genres for the filter widget (genres with ≥ 200 tracks)
TOP_GENRES = sorted(
    df["genre"].value_counts().loc[lambda s: s >= 200].index.tolist()
)

# ──────────────────────────────────────────────────────────────────────────────
# Emotion → Valence / Arousal mapping  (Russell's circumplex model)
# ──────────────────────────────────────────────────────────────────────────────
EMOTION_PROFILES = {
    "Happy":    {"val_lo": 5.8, "val_hi": 9.0, "aro_lo": 4.3, "aro_hi": 9.0},
    "Sad":      {"val_lo": 0.0, "val_hi": 4.2, "aro_lo": 0.0, "aro_hi": 3.8},
    "Angry":    {"val_lo": 0.0, "val_hi": 4.5, "aro_lo": 5.0, "aro_hi": 9.0},
    "Fearful":  {"val_lo": 0.0, "val_hi": 4.5, "aro_lo": 3.0, "aro_hi": 5.5},
    "Neutral":  {"val_lo": 4.0, "val_hi": 6.8, "aro_lo": 3.2, "aro_hi": 5.2},
    "Surprised":{"val_lo": 4.5, "val_hi": 8.0, "aro_lo": 5.0, "aro_hi": 9.0},
    "Disgusted":{"val_lo": 0.0, "val_hi": 3.8, "aro_lo": 3.5, "aro_hi": 6.0},
}

# Emotion badge colours
EMOTION_COLORS = {
    "Happy": "#FFD700",
    "Sad": "#4A90D9",
    "Angry": "#E74C3C",
    "Fearful": "#9B59B6",
    "Neutral": "#95A5A6",
    "Surprised": "#F39C12",
    "Disgusted": "#27AE60",
}

# Canonical label list (model output order)
EMOTION_LABELS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]


def recommend_songs(emotions: list[str], genre_filter: list[str] | None = None, n: int = 20) -> pd.DataFrame:
    """
    Return *n* song recommendations by filtering the dataset through
    valence / arousal ranges that correspond to the detected emotions.

    Emotions earlier in the list are weighted more heavily (they were
    detected more frequently).
    """
    if not emotions:
        return pd.DataFrame()

    source = df.copy()
    if genre_filter:
        source = source[source["genre"].isin(genre_filter)]
        if source.empty:
            return pd.DataFrame()

    # Weight distribution — primary emotion gets the most tracks
    weights = _distribute_weights(len(emotions), n)
    parts: list[pd.DataFrame] = []
    for emo, count in zip(emotions, weights):
        profile = EMOTION_PROFILES.get(emo)
        if profile is None:
            continue
        subset = source[
            (source["pleasant"] >= profile["val_lo"])
            & (source["pleasant"] <= profile["val_hi"])
            & (source["arousal"] >= profile["aro_lo"])
            & (source["arousal"] <= profile["aro_hi"])
        ]
        if subset.empty:
            continue
        sample_n = min(count, len(subset))
        parts.append(subset.sample(n=sample_n))

    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["name", "artist"])
    return result.head(n)


def _distribute_weights(num_emotions: int, total: int) -> list[int]:
    """Distribute *total* across *num_emotions* with decreasing weight."""
    if num_emotions == 0:
        return []
    if num_emotions == 1:
        return [total]
    # Ratio: 50%, 25%, 15%, 10% ... (geometric-ish)
    raw = [max(1, int(total * (0.5 ** i))) for i in range(num_emotions)]
    scale = total / max(sum(raw), 1)
    adjusted = [max(1, int(r * scale)) for r in raw]
    # Fix rounding
    diff = total - sum(adjusted)
    adjusted[0] += diff
    return adjusted


def rank_emotions(detected: list[str]) -> list[str]:
    """Return unique emotions ordered by descending frequency."""
    if not detected:
        return []
    counts = Counter(detected)
    return [e for e, _ in counts.most_common()]


# ── CNN Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading emotion recognition model …")
def load_model():
    return load_model(str(BASE_DIR / "model.h5"))


model = load_model()

# ── Haar Cascade ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cascade():
    cv2.ocl.setUseOpenCL(False)
    c = cv2.CascadeClassifier(str(BASE_DIR / "haarcascade_frontalface_default.xml"))
    if c.empty():
        st.error("Failed to load Haar cascade classifier.")
    return c


face_cascade = load_cascade()


# ── Favorites helpers ────────────────────────────────────────────────────────
def add_to_favorites(name: str, artist: str, web_link: str, youtube: str):
    for f in st.session_state["favorites"]:
        if f.get("name") == name and f.get("artist") == artist:
            return
    st.session_state["favorites"].append(
        {"name": name, "artist": artist, "web_link": web_link, "youtube": youtube}
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium Dark Glassmorphism
# ══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg-primary:    #0a0a14;
  --bg-secondary:  #12121f;
  --bg-card:       rgba(255, 255, 255, 0.04);
  --border-card:   rgba(255, 255, 255, 0.08);
  --text-primary:  #eaeaf4;
  --text-muted:    #8b8fa8;
  --accent-1:      #7c5cff;
  --accent-2:      #22c1c3;
  --accent-gradient: linear-gradient(135deg, #7c5cff 0%, #22c1c3 100%);
  --danger:        #e74c3c;
  --success:       #27ae60;
}

html, body, .stApp {
  background: linear-gradient(160deg, var(--bg-primary) 0%, var(--bg-secondary) 50%, #0d0f1a 100%) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(10, 10, 20, 0.95) !important;
  border-right: 1px solid var(--border-card);
}
section[data-testid="stSidebar"] .stRadio > label {
  color: var(--text-muted) !important;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  font-size: 11px;
}

/* Headers */
.hero-header {
  text-align: center;
  padding: 24px 0 8px 0;
}
.hero-title {
  font-weight: 900;
  font-size: 2.4rem;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.5px;
  margin: 0;
  line-height: 1.2;
}
.hero-subtitle {
  color: var(--text-muted);
  font-size: 15px;
  margin-top: 6px;
  font-weight: 400;
}

/* Glass panels */
.glass-panel {
  background: var(--bg-card);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border-card);
  border-radius: 18px;
  padding: 22px 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  margin-bottom: 16px;
}
.glass-panel-sm {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 10px;
}

/* Song cards */
.song-card {
  display: flex;
  gap: 14px;
  align-items: center;
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  padding: 14px 18px;
  border-radius: 14px;
  margin-bottom: 8px;
  transition: all 0.25s ease;
}
.song-card:hover {
  border-color: rgba(124, 92, 255, 0.35);
  box-shadow: 0 4px 20px rgba(124, 92, 255, 0.12);
  transform: translateY(-1px);
}
.song-rank {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-gradient);
  color: #fff; font-weight: 800; font-size: 14px;
  flex-shrink: 0;
}
.song-info { flex: 1; min-width: 0; }
.song-name {
  color: var(--text-primary);
  font-weight: 700;
  font-size: 14px;
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.song-name a { color: var(--text-primary); text-decoration: none; }
.song-name a:hover { color: var(--accent-2); }
.song-artist {
  color: var(--text-muted);
  font-size: 12px;
  margin: 2px 0 0 0;
}
.song-genre-badge {
  display: inline-block;
  background: rgba(124, 92, 255, 0.15);
  color: var(--accent-1);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 4px;
}
.song-links a {
  color: var(--text-muted);
  font-size: 12px;
  margin-left: 10px;
  text-decoration: none;
  transition: color 0.2s;
}
.song-links a:hover { color: var(--accent-2); }

/* Emotion badge */
.emotion-badge {
  display: inline-block;
  padding: 5px 16px;
  border-radius: 24px;
  font-weight: 700;
  font-size: 13px;
  margin: 3px 4px;
  border: 1px solid rgba(255,255,255,0.1);
}

/* Stats card */
.stat-card {
  text-align: center;
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: 14px;
  padding: 18px 12px;
}
.stat-value {
  font-size: 2rem;
  font-weight: 800;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.stat-label {
  color: var(--text-muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-top: 4px;
}

/* Section headers */
.section-header {
  font-weight: 700;
  font-size: 18px;
  color: var(--text-primary);
  margin: 28px 0 14px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-card);
}

/* Buttons */
.stButton > button {
  background: var(--accent-gradient) !important;
  color: #fff !important;
  border: 0 !important;
  border-radius: 10px !important;
  padding: 8px 20px !important;
  font-weight: 700 !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  filter: brightness(1.1) !important;
  box-shadow: 0 4px 15px rgba(124, 92, 255, 0.35) !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
  background: var(--accent-gradient) !important;
}

/* Expanders */
.streamlit-expanderHeader {
  font-weight: 600 !important;
  color: var(--text-primary) !important;
}

/* Plotly chart background */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Footer */
.app-footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
  padding: 30px 0 10px 0;
  border-top: 1px solid var(--border-card);
  margin-top: 40px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Navigation & Controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
      <span style="font-size: 40px;">🎵</span>
      <div style="font-weight:800; font-size:22px; background: linear-gradient(135deg,#7c5cff,#22c1c3);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-top:4px;">
        Moodify
      </div>
      <div style="color:#8b8fa8; font-size:12px; margin-top:2px;">Emotion-Based Music AI</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🎵 Emotion Scanner", "📊 Analytics Dashboard", "ℹ️ About & Methodology"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Genre filter (available on all pages but primarily for Scanner)
    st.markdown("**🎸 Genre Filter**")
    genre_all = st.checkbox("All genres", value=True)
    if genre_all:
        selected_genres = []
    else:
        selected_genres = st.multiselect(
            "Select genres",
            TOP_GENRES,
            default=["pop", "rock", "indie", "electronic"],
            label_visibility="collapsed",
        )

    st.markdown("---")
    num_songs = st.slider("Songs to recommend", 5, 50, 20)

    st.markdown("---")
    st.markdown(
        "<div style='color:#555; font-size:11px; text-align:center;'>"
        "Built with Streamlit · TF/Keras<br>© 2026 Moodify Project</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Emotion Scanner
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎵 Emotion Scanner":

    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">Emotion-Based Music Recommendation</div>
      <div class="hero-subtitle">Scan your face to detect emotion, then get personalised song recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scanner Controls ─────────────────────────────────────────────────
    scan_col, info_col = st.columns([3, 2], gap="large")

    detected_emotions: list[str] = []
    prediction_accumulator: list[np.ndarray] = []

    with scan_col:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### 📷 Live Emotion Scanner")
        analysis_seconds = st.slider(
            "Analysis duration (seconds)", min_value=2, max_value=10, value=5
        )

        if st.button("🔍  Start Scan", use_container_width=True):
            detected_emotions.clear()
            prediction_accumulator.clear()

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                st.error("❌ Unable to access the camera. Check permissions or device.")
                st.stop()

            # Warmup
            warmup_t = time.time()
            while time.time() - warmup_t < 0.5:
                cap.read()
                time.sleep(0.01)

            # Countdown
            status_ph = st.empty()
            for sec in range(3, 0, -1):
                status_ph.info(f"🎬 Starting in **{sec}s** — look at the camera …")
                time.sleep(1)
            status_ph.empty()

            frame_ph = st.empty()
            progress_ph = st.progress(0)
            start_time = time.time()
            duration = float(analysis_seconds)
            stride = 3
            frame_count = 0
            last_label = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                h, w = gray.shape[:2]
                min_face = (max(40, int(w * 0.12)), max(40, int(h * 0.12)))
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5,
                    flags=cv2.CASCADE_SCALE_IMAGE, minSize=min_face,
                )
                frame_count += 1

                if len(faces) > 0:
                    target = max(faces, key=lambda b: b[2] * b[3])
                    (x, y, fw, fh) = target
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y - 50), (x + fw, y + fh + 10), (124, 92, 255), 2)

                    if frame_count % stride == 0:
                        roi = gray[y:y + fh, x:x + fw]
                        face48 = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
                        face48 = face48.astype("float32") / 255.0
                        inp = np.expand_dims(np.expand_dims(face48, -1), 0)
                        pred = model.predict(inp, verbose=0)[0]
                        prediction_accumulator.append(pred)
                        max_idx = int(np.argmax(pred))
                        conf = float(np.max(pred))
                        if conf >= 0.45:
                            last_label = EMOTION_LABELS[max_idx]
                            detected_emotions.append(last_label)

                    if last_label:
                        cv2.putText(
                            frame, last_label, (x + 20, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA,
                        )

                frame_ph.image(
                    cv2.resize(frame, (720, 500), interpolation=cv2.INTER_CUBIC),
                    channels="BGR",
                )
                elapsed = time.time() - start_time
                progress_ph.progress(min(100, int(elapsed / duration * 100)))
                if elapsed >= duration:
                    break
                time.sleep(0.01)

            cap.release()
            progress_ph.empty()

            # Save to session state
            ranked = rank_emotions(detected_emotions)
            st.session_state["last_emotions"] = ranked
            if prediction_accumulator:
                avg_pred = np.mean(prediction_accumulator, axis=0)
                st.session_state["last_predictions"] = avg_pred
            else:
                st.session_state["last_predictions"] = None

            if ranked:
                st.success(f"✅ Detected emotions: **{', '.join(ranked)}**")
            else:
                st.warning("⚠️ No face/emotion detected. Try better lighting or move closer.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Manual mood override
        st.markdown('<div class="glass-panel-sm">', unsafe_allow_html=True)
        st.markdown("**🎛️ Manual Mood Override** *(optional)*")
        manual_moods = st.multiselect(
            "Select moods manually",
            EMOTION_LABELS,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Confidence Visualisation ────────────────────────────────────────
    with info_col:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("#### 📊 Emotion Confidence")

        preds = st.session_state.get("last_predictions")
        if preds is not None:
            pred_df = pd.DataFrame({
                "Emotion": EMOTION_LABELS,
                "Confidence": (preds * 100).round(1),
            }).sort_values("Confidence", ascending=True)

            fig = px.bar(
                pred_df, x="Confidence", y="Emotion",
                orientation="h",
                color="Confidence",
                color_continuous_scale=["#1a1a2e", "#7c5cff", "#22c1c3"],
                text="Confidence",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#eaeaf4"),
                xaxis=dict(title="Confidence (%)", gridcolor="rgba(255,255,255,0.05)", range=[0, 105]),
                yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                coloraxis_showscale=False,
                height=340,
                margin=dict(l=0, r=20, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Scan your face to see the CNN confidence distribution across all 7 emotions.")

        # Show detected emotion badges
        last_emos = st.session_state.get("last_emotions", [])
        if last_emos:
            badge_html = '<div style="margin-top:8px;">'
            for e in last_emos:
                c = EMOTION_COLORS.get(e, "#7c5cff")
                badge_html += (
                    f'<span class="emotion-badge" '
                    f'style="background:rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.18); '
                    f'color:{c};">{e}</span>'
                )
            badge_html += "</div>"
            st.markdown(badge_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Determine active emotions ────────────────────────────────────────
    active_emotions = manual_moods if manual_moods else st.session_state.get("last_emotions", [])

    # ── Recommendations ──────────────────────────────────────────────────
    if active_emotions:
        rec_df = recommend_songs(
            active_emotions,
            genre_filter=selected_genres if selected_genres else None,
            n=num_songs,
        )

        # Log mood history
        if not rec_df.empty:
            st.session_state["mood_history"].append({
                "timestamp": int(time.time()),
                "moods": ", ".join(active_emotions),
                "top_song": rec_df.iloc[0]["name"],
                "top_artist": rec_df.iloc[0]["artist"],
            })

        # Mood/Energy stats
        if not rec_df.empty:
            st.markdown('<div class="section-header">📈 Batch Stats</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{rec_df["pleasant"].mean():.2f}</div>'
                    f'<div class="stat-label">Avg Pleasantness</div></div>',
                    unsafe_allow_html=True,
                )
            with s2:
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value">{rec_df["arousal"].mean():.2f}</div>'
                    f'<div class="stat-label">Avg Energy</div></div>',
                    unsafe_allow_html=True,
                )
            with s3:
                genre_mode = rec_df["genre"].mode()
                top_genre = genre_mode.iloc[0] if not genre_mode.empty else "—"
                st.markdown(
                    f'<div class="stat-card"><div class="stat-value" style="font-size:1.3rem;">{top_genre}</div>'
                    f'<div class="stat-label">Top Genre</div></div>',
                    unsafe_allow_html=True,
                )

        # Song list
        st.markdown(
            '<div class="section-header">🎶 Recommended Songs</div>',
            unsafe_allow_html=True,
        )

        if not rec_df.empty:
            fav_ids = {(f["name"], f["artist"]) for f in st.session_state["favorites"]}

            for i, row in rec_df.iterrows():
                idx = rec_df.index.get_loc(i) + 1
                yt = to_youtube_search(row["name"], row["artist"])
                genre_tag = row.get("genre", "")
                genre_badge = f'<span class="song-genre-badge">{genre_tag}</span>' if genre_tag else ""

                card_cols = st.columns([0.94, 0.06])
                with card_cols[0]:
                    st.markdown(
                        f"""<div class="song-card">
                          <div class="song-rank">{idx}</div>
                          <div class="song-info">
                            <p class="song-name"><a href="{row['web_link']}" target="_blank">{row['name']}</a></p>
                            <p class="song-artist">{row['artist']}</p>
                            {genre_badge}
                          </div>
                          <div class="song-links">
                            <a href="{row['web_link']}" target="_blank">Spotify</a>
                            <a href="{yt}" target="_blank">YouTube</a>
                          </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with card_cols[1]:
                    key = f"fav_{hash((row['name'], row['artist']))}_{idx}"
                    if (row["name"], row["artist"]) in fav_ids:
                        st.button("★", key=key, disabled=True, help="In Favorites")
                    else:
                        st.button(
                            "☆", key=key, help="Add to Favorites",
                            on_click=add_to_favorites,
                            args=(row["name"], row["artist"], row["web_link"], yt),
                        )

            # Export
            csv_bytes = rec_df[["name", "artist", "genre", "web_link"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  Export Recommendations as CSV",
                csv_bytes,
                file_name="moodify_recommendations.csv",
                mime="text/csv",
            )
        else:
            st.info("No songs matched these filters. Try broadening your genre selection or adjusting emotions.")

    # ── Favorites ────────────────────────────────────────────────────────
    with st.expander("⭐ My Favorites"):
        fav_df = pd.DataFrame(st.session_state["favorites"])
        if not fav_df.empty:
            for _, row in fav_df.iterrows():
                st.markdown(
                    f"- **{row['name']}** by *{row['artist']}* "
                    f"— [Spotify]({row['web_link']}) · [YouTube]({row['youtube']})",
                    unsafe_allow_html=True,
                )
            fav_csv = fav_df[["name", "artist", "web_link", "youtube"]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Export Favorites", fav_csv, file_name="moodify_favorites.csv", mime="text/csv")
        else:
            st.info("No favorites yet — click ☆ on any song to save it.")

    # ── Mood Journal ─────────────────────────────────────────────────────
    with st.expander("📖 Mood Journal"):
        hist_df = pd.DataFrame(st.session_state["mood_history"])
        if not hist_df.empty:
            hist_df["datetime"] = pd.to_datetime(hist_df["timestamp"], unit="s")
            st.dataframe(
                hist_df[["datetime", "moods", "top_song", "top_artist"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No scans logged yet.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics Dashboard":

    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">Analytics Dashboard</div>
      <div class="hero-subtitle">Visualise your emotion scanning history and mood patterns</div>
    </div>
    """, unsafe_allow_html=True)

    hist = st.session_state["mood_history"]

    if not hist:
        st.info("No scan data yet. Use the Emotion Scanner to start building your analytics.")
    else:
        hist_df = pd.DataFrame(hist)
        hist_df["datetime"] = pd.to_datetime(hist_df["timestamp"], unit="s")

        # ── Summary Stats ────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        all_moods = hist_df["moods"].str.split(", ").explode()
        with c1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{len(hist_df)}</div>'
                f'<div class="stat-label">Total Scans</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            top_mood = all_moods.value_counts().idxmax() if not all_moods.empty else "—"
            st.markdown(
                f'<div class="stat-card"><div class="stat-value" style="font-size:1.6rem;">{top_mood}</div>'
                f'<div class="stat-label">Most Frequent Mood</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            unique_moods = all_moods.nunique()
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{unique_moods}</div>'
                f'<div class="stat-label">Unique Emotions</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            unique_songs = hist_df["top_song"].nunique()
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{unique_songs}</div>'
                f'<div class="stat-label">Unique Top Songs</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Charts ───────────────────────────────────────────────────────
        chart_l, chart_r = st.columns(2, gap="large")

        with chart_l:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("##### Mood Distribution")
            mood_counts = all_moods.value_counts().reset_index()
            mood_counts.columns = ["Emotion", "Count"]
            emo_colors = [EMOTION_COLORS.get(e, "#7c5cff") for e in mood_counts["Emotion"]]
            fig_pie = px.pie(
                mood_counts, names="Emotion", values="Count",
                color_discrete_sequence=emo_colors,
                hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#eaeaf4"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
                legend=dict(font=dict(size=12)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_r:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown("##### Mood Timeline")
            timeline_df = hist_df[["datetime", "moods"]].copy()
            timeline_df["primary_mood"] = timeline_df["moods"].str.split(", ").str[0]
            fig_tl = px.scatter(
                timeline_df, x="datetime", y="primary_mood",
                color="primary_mood",
                color_discrete_map=EMOTION_COLORS,
                size_max=12,
            )
            fig_tl.update_traces(marker=dict(size=14, line=dict(width=1, color="#1a1a2e")))
            fig_tl.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#eaeaf4"),
                xaxis=dict(title="Time", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False,
                height=320,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig_tl, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Full History Table ───────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Scan History</div>', unsafe_allow_html=True)
        st.dataframe(
            hist_df[["datetime", "moods", "top_song", "top_artist"]].sort_values("datetime", ascending=False),
            hide_index=True,
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — About & Methodology
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About & Methodology":

    st.markdown("""
    <div class="hero-header">
      <div class="hero-title">About & Methodology</div>
      <div class="hero-subtitle">Technical details of the Moodify system</div>
    </div>
    """, unsafe_allow_html=True)

    # ── System Architecture ──────────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
    │  Webcam      │────▶│ Haar Cascade     │────▶│ FER CNN (48×48 px)   │────▶│ Emotion Labels     │
    │  (Live Feed) │     │ Face Detection   │     │ 7-class Softmax      │     │ + Confidence Scores│
    └─────────────┘     └──────────────────┘     └──────────────────────┘     └────────┬───────────┘
                                                                                       │
                                                                                       ▼
    ┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
    │ Song Cards   │◀────│ Weighted Sampling │◀────│ Valence / Arousal    │◀────│ Russell's          │
    │ Spotify / YT │     │ Genre Filtering   │     │ Range Filtering      │     │ Circumplex Model   │
    └─────────────┘     └──────────────────┘     └──────────────────────┘     └────────────────────┘
    ```
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── CNN Architecture ─────────────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 CNN Model Architecture")
    st.markdown("""
    The emotion recognition model is a **Convolutional Neural Network** trained on the
    **FER-2013** dataset (35,887 grayscale 48×48 images across 7 emotion classes).

    | Layer | Type | Parameters |
    |-------|------|-----------|
    | 1 | Conv2D (32 filters, 3×3) + ReLU | 320 |
    | 2 | Conv2D (64 filters, 3×3) + ReLU | 18,496 |
    | 3 | MaxPooling2D (2×2) | — |
    | 4 | Conv2D (128 filters, 3×3) + ReLU | 73,856 |
    | 5 | MaxPooling2D (2×2) | — |
    | 6 | Conv2D (128 filters, 3×3) + ReLU | 147,584 |
    | 7 | MaxPooling2D (2×2) | — |
    | 8 | Dropout (0.25) | — |
    | 9 | Flatten | — |
    | 10 | Dense (1024) + ReLU | 2,098,176 |
    | 11 | Dropout (0.5) | — |
    | 12 | Dense (7) + Softmax | 7,175 |

    **Total parameters: ~2.35M** · **Output: 7-class probability distribution**

    **Classes:** Angry · Disgusted · Fearful · Happy · Neutral · Sad · Surprised
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Emotion → Music Mapping ──────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Emotion → Music Mapping (Russell's Circumplex Model)")
    st.markdown("""
    Music recommendations are based on **Russell's circumplex model of affect**,
    which maps emotions onto two dimensions:

    - **Valence** (pleasantness): low = negative/unpleasant → high = positive/pleasant
    - **Arousal** (energy): low = calm/subdued → high = excited/intense

    Each detected emotion is mapped to a valence × arousal range, and songs whose
    tags fall within that range are sampled:
    """)

    mapping_data = []
    for emo, prof in EMOTION_PROFILES.items():
        mapping_data.append({
            "Emotion": emo,
            "Valence Range": f"{prof['val_lo']:.1f} – {prof['val_hi']:.1f}",
            "Arousal Range": f"{prof['aro_lo']:.1f} – {prof['aro_hi']:.1f}",
            "Colour": EMOTION_COLORS.get(emo, "#777"),
        })
    st.table(pd.DataFrame(mapping_data)[["Emotion", "Valence Range", "Arousal Range"]])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📦 MUSE v3 Dataset")
    st.markdown(f"""
    The music metadata comes from the **MUSE (MUsic and Sentiment/Emotion) v3** dataset:

    | Metric | Value |
    |--------|-------|
    | Total tracks | {len(df):,} |
    | Unique genres | {df['genre'].nunique()} |
    | Valence range | {df['pleasant'].min():.2f} – {df['pleasant'].max():.2f} |
    | Arousal range | {df['arousal'].min():.2f} – {df['arousal'].max():.2f} |
    | Features per track | name, artist, genre, valence, arousal, dominance |

    Each track is annotated with crowd-sourced emotion tags, and continuous
    valence / arousal / dominance scores derived from those tags.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Tech Stack ───────────────────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Technology Stack")
    st.markdown("""
    | Component | Technology |
    |-----------|-----------|
    | Web Framework | Streamlit |
    | Deep Learning | TensorFlow / Keras |
    | Computer Vision | OpenCV (Haar Cascade) |
    | Data Processing | Pandas, NumPy |
    | Visualisation | Plotly |
    | Dataset | MUSE v3 (90K tracks) |
    | Language | Python 3.10+ |
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── References ───────────────────────────────────────────────────────
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📚 References")
    st.markdown("""
    1. Goodfellow, I. J. et al. (2013). *Challenges in Representation Learning:
       A Report on Three Machine Learning Contests.* Neural Information Processing, ICONIP 2013.
       (FER-2013 dataset)
    2. Russell, J. A. (1980). *A Circumplex Model of Affect.* Journal of Personality
       and Social Psychology, 39(6), 1161–1178.
    3. Cano, E. & Morisio, M. (2017). *MoodyLyrics: A Sentiment Annotated Lyrics Dataset.*
       Proceedings of the 2017 ACM Intelligent Systems and Technologies Conference.
    4. Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade
       of Simple Features.* CVPR 2001.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  🎵 <strong>Moodify</strong> — Emotion-Based Music Recommendation System<br>
  Built with Streamlit · TensorFlow/Keras · OpenCV · MUSE v3 Dataset<br>
  Academic Project © 2026
</div>
""", unsafe_allow_html=True)
