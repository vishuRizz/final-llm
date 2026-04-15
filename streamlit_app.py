"""
MiniLLM – Streamlit Chat Interface
A tiny decoder-only Transformer you can train and chat with in the browser.
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st
import torch

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "tiny_llm.pt"
TOKENIZER_FILE = ARTIFACTS_DIR / "tokenizer.model"
CONFIG_FILE = ARTIFACTS_DIR / "training_config.json"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MiniLLM · Tiny Transformer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero gradient header */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 179, 237, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #63b3ed, #a78bfa, #f687b3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    }
    .hero p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1.1rem;
        margin: 0.4rem 0 0.4rem 3rem;
        font-size: 0.95rem;
        line-height: 1.5;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    }
    .chat-assistant {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: #e2e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1.1rem;
        margin: 0.4rem 3rem 0.4rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
        word-wrap: break-word;
        border: 1px solid rgba(99,179,237,0.15);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .role-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }
    .role-user { color: #93c5fd; }
    .role-assistant { color: #a78bfa; }

    /* Info cards */
    .info-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .info-card b { color: #63b3ed; }

    /* Status badges */
    .badge-ready {
        display: inline-block;
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        letter-spacing: 0.05em;
    }
    .badge-missing {
        display: inline-block;
        background: linear-gradient(90deg, #dc2626, #f87171);
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        letter-spacing: 0.05em;
    }

    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def model_is_ready() -> bool:
    return MODEL_FILE.exists() and TOKENIZER_FILE.exists()

@st.cache_resource(show_spinner=False)
def load_inference_engine():
    """Load model + tokenizer once and cache across reruns."""
    from src.mini_llm.infer import load_model_and_tokenizer
    return load_model_and_tokenizer()

def stream_reply(prompt: str, max_new_tokens: int, temperature: float, top_k: int):
    """Yield tokens one-by-one for a streaming chat feel."""
    import torch
    from src.mini_llm.tokenizer_utils import decode_ids, encode_text
    from src.mini_llm.infer import BLOCK_SIZE

    model, sp, device = load_inference_engine()

    wrapped = f"<|user|> {prompt}\n<|assistant|>"
    token_ids = encode_text(sp, wrapped)
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    generated_ids = []

    for _ in range(max_new_tokens):
        x_cond = x[:, -BLOCK_SIZE:] if x.size(1) > BLOCK_SIZE else x
        logits = model(x_cond)[:, -1, :]
        logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        
        generated_ids.append(next_id.item())
        current_text = decode_ids(sp, generated_ids)
        
        if "<|user|>" in current_text or "<|assistant|>" in current_text:
            current_text = current_text.replace("<|user|>", "").replace("<|assistant|>", "").strip()
            yield current_text
            break
            
        yield current_text

# ── UI Layout ─────────────────────────────────────────────────────────────────

st.markdown(
    '''
    <div class="hero">
        <h1>MiniLLM Chat</h1>
        <p>A tiny custom-built Transformer trained from scratch, ready to converse.</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8636/8636901.png", width=64)
    st.markdown("### Model Status")

    if model_is_ready():
        st.markdown('<div class="badge-ready">● Ready</div>', unsafe_allow_html=True)
        
        # Attempt to load config for display
        epochs_run = "?"
        loss_val = "?"
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text())
                epochs_run = cfg.get('epochs', '?')
                loss_val = cfg.get('final_loss', '?')
                if isinstance(loss_val, float):
                    loss_val = f"{loss_val:.4f}"
            except:
                pass
                
        st.markdown(f"""
        <div class="info-card" style="margin-top: 1rem;">
            <div><b>Artifacts</b>: found</div>
            <div><b>Epochs</b>: {epochs_run}</div>
            <div><b>Loss</b>: {loss_val}</div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="badge-missing">● Missing Artifacts</div>', unsafe_allow_html=True)
        st.info("Train the model first using the training scripts, or copy artifacts into the `artifacts/` folder.")

    st.markdown("---")
    st.markdown("### Inference Settings")
    max_tokens = st.slider("Max tokens", 10, 200, 80)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.8, 0.1)
    top_k = st.slider("Top K", 0, 100, 40)

# Main chat flow
if not model_is_ready():
    st.warning("Cannot start chat. Model artifacts are missing.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history safely
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'''
        <div class="role-label role-user">You</div>
        <div class="chat-user">{msg["content"]}</div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="role-label role-assistant">MiniLLM</div>
        <div class="chat-assistant">{msg["content"]}</div>
        ''', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.markdown(f'''
    <div class="role-label role-user">You</div>
    <div class="chat-user">{prompt}</div>
    ''', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Stream assistant response
    st.markdown('<div class="role-label role-assistant" style="margin-top:1rem;">MiniLLM</div>', unsafe_allow_html=True)
    
    reply_container = st.empty()
    full_reply = ""
    
    with st.spinner("Thinking..."):
        try:
            for current_text in stream_reply(prompt, max_tokens, temperature, top_k):
                full_reply = current_text
                reply_container.markdown(f'<div class="chat-assistant">{full_reply}▌</div>', unsafe_allow_html=True)
                time.sleep(0.01)  # small pause for visual effect
            
            # Final render without cursor
            reply_container.markdown(f'<div class="chat-assistant">{full_reply}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})
            
        except Exception as e:
            st.error(f"Error during inference: {e}")
