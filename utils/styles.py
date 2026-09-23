"""Premium fintech CSS theme: glassmorphism cards, gradients, and the
requested color palette, injected into Streamlit via st.markdown."""

PALETTE = {
    "primary_dark": "#37315F",
    "secondary_purple": "#544A84",
    "accent_gold": "#F8C685",
    "accent_coral": "#F6A696",
    "background": "#FDFCFB",
    "card": "#FFFFFF",
}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Times New Roman', Times, serif;
    }}

    .stApp {{
        background: {PALETTE['background']};
    }}

    /* Hero / gradient banner */
    .hero-banner {{
        background: linear-gradient(135deg, {PALETTE['primary_dark']} 0%, {PALETTE['secondary_purple']} 100%);
        border-radius: 28px;
        padding: 3rem 2.5rem;
        color: white;
        box-shadow: 0 20px 60px rgba(55, 49, 95, 0.35);
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
    }}
    .hero-banner h1 {{
       font-family: 'Times New Roman', Times, serif;
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #FFFFFF 0%, {PALETTE['accent_gold']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-banner p {{
        font-size: 1.1rem;
        opacity: 0.9;
        max-width: 640px;
    }}

    /* Glassmorphism cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 32px rgba(55, 49, 95, 0.10);
        margin-bottom: 1rem;
    }}

    .metric-card {{
        background: {PALETTE['card']};
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 6px 18px rgba(55, 49, 95, 0.08);
        border-left: 4px solid {PALETTE['secondary_purple']};
    }}
    .metric-card .label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b6483;
        font-weight: 600;
    }}
    .metric-card .value {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: {PALETTE['primary_dark']};
    }}

    .badge-bullish {{
        background: rgba(120, 191, 149, 0.15);
        color: #2f9e5b;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-bearish {{
        background: rgba(246, 166, 150, 0.20);
        color: #c65a45;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .badge-neutral {{
        background: rgba(248, 198, 133, 0.25);
        color: #a06a1f;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }}

    /* Section headers */
    .section-title {{
        font-family: 'Times New Roman', Times, serif;
        font-weight: 700;
        font-size: 1.3rem;
        color: {PALETTE['primary_dark']};
        margin: 1.2rem 0 0.6rem 0;
        border-left: 5px solid {PALETTE['accent_gold']};
        padding-left: 0.7rem;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {PALETTE['primary_dark']}, {PALETTE['secondary_purple']});
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1.3rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(55, 49, 95, 0.3);
    }}

    /* Ticker strip */
    .ticker-strip {{
        display: flex;
        gap: 1.4rem;
        overflow-x: auto;
        padding: 0.8rem 0;
    }}
    .ticker-pill {{
        background: {PALETTE['card']};
        border-radius: 14px;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 14px rgba(55, 49, 95, 0.08);
        white-space: nowrap;
        font-weight: 600;
        font-size: 0.9rem;
    }}

    footer {{visibility: hidden;}}
</style>
"""


def metric_card_html(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


def trend_badge_html(trend: str) -> str:
    cls = {"Bullish": "badge-bullish", "Bearish": "badge-bearish", "Neutral": "badge-neutral"}.get(trend, "badge-neutral")
    return f'<span class="{cls}">{trend}</span>'
