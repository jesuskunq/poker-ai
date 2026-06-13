"""
Poker Decision AI — Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle
import os
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Poker Decision AI",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a1628; color: #e8dcc8; }
h1, h2, h3 { font-family: 'Playfair Display', serif; color: #c9a84c; }

.decision-box { border-radius: 12px; padding: 28px 32px; text-align: center; font-family: 'Playfair Display', serif; margin: 16px 0; }
.fold-box  { background: linear-gradient(135deg,#1a0a0a,#2d1111); border: 2px solid #ef4444; }
.call-box  { background: linear-gradient(135deg,#0a1a0a,#112d11); border: 2px solid #22c55e; }
.raise-box { background: linear-gradient(135deg,#1a140a,#2d2311); border: 2px solid #c9a84c; }
.decision-label { font-size: 2.4rem; font-weight: 700; letter-spacing: 2px; }
.fold-box  .decision-label { color: #ef4444; }
.call-box  .decision-label { color: #22c55e; }
.raise-box .decision-label { color: #c9a84c; }

.prob-bar-wrap { margin: 8px 0; }
.prob-label { font-size: 12px; font-family: 'DM Mono', monospace; display: flex; justify-content: space-between; margin-bottom: 3px; }
.prob-track { height: 8px; background: #1a2a3a; border-radius: 4px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 4px; }
.fold-fill  { background: #ef4444; }
.call-fill  { background: #22c55e; }
.raise-fill { background: #c9a84c; }

.stat-box { background: #0d1f3c; border: 1px solid #1e3a5f; border-radius: 8px; padding: 14px 16px; text-align: center; }
.stat-val { font-size: 1.5rem; font-family: 'DM Mono', monospace; color: #c9a84c; font-weight: 500; }
.stat-lbl { font-size: 11px; color: #6b7a99; margin-top: 2px; }

.hand-display { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.big-card { width:56px; height:76px; background:#f5f0e8; border-radius:8px; border:1px solid #c9a84c; display:flex; flex-direction:column; align-items:center; justify-content:center; font-size:18px; font-weight:700; font-family:'DM Mono',monospace; color:#1a1a1a; box-shadow:2px 3px 8px rgba(0,0,0,0.4); }
.big-card.red   { color: #dc2626; }
.big-card.empty { background:#0d1f3c; border:2px dashed #2a3a5c; color:#2a3a5c; font-size:22px; }

.section-title { font-family:'DM Mono',monospace; font-size:11px; text-transform:uppercase; letter-spacing:2px; color:#6b7a99; margin-bottom:8px; }
.ev-insight { background:#0d1f3c; border-left:3px solid #c9a84c; border-radius:0 8px 8px 0; padding:12px 16px; font-size:13px; color:#b8c9e0; margin-top:12px; }

.wl-box { background: #0d1f3c; border: 1px solid #1e3a5f; border-radius: 12px; padding: 18px 20px; margin-top: 14px; }
.wl-title { font-family:'DM Mono',monospace; font-size:11px; text-transform:uppercase; letter-spacing:2px; color:#6b7a99; margin-bottom:12px; }
.wl-bar-row { display:flex; height:28px; border-radius:6px; overflow:hidden; margin: 10px 0 6px; }
.wl-win  { background:#22c55e; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; color:#fff; font-family:'DM Mono',monospace; }
.wl-tie  { background:#c9a84c; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; color:#fff; font-family:'DM Mono',monospace; }
.wl-lose { background:#ef4444; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; color:#fff; font-family:'DM Mono',monospace; }
.wl-legend { display:flex; gap:16px; font-size:11px; font-family:'DM Mono',monospace; color:#6b7a99; margin-top:6px; }
.wl-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }

.raise-size-box { background:#0d1f3c; border:1px solid #c9a84c; border-radius:10px; padding:14px 18px; margin-top:10px; display:flex; align-items:center; gap:16px; }
.raise-size-amount { font-family:'DM Mono',monospace; font-size:1.6rem; font-weight:600; color:#c9a84c; }
.raise-size-detail { font-size:12px; color:#6b7a99; line-height:1.6; }

.replay-box { background:#0a1628; border:1px solid #1e3a5f; border-radius:12px; padding:20px; margin-top:8px; }
.replay-street-row { display:flex; align-items:flex-start; gap:14px; padding:12px 0; border-bottom:1px solid #1a2a3a; }
.replay-street-row:last-child { border-bottom: none; }
.replay-street-badge { font-family:'DM Mono',monospace; font-size:10px; text-transform:uppercase; letter-spacing:1px; background:#1a2a3a; color:#6b7a99; padding:3px 8px; border-radius:4px; white-space:nowrap; min-width:60px; text-align:center; margin-top:2px; }
.replay-cards { display:flex; gap:4px; flex-wrap:wrap; }
.sm-card { width:34px; height:46px; background:#f5f0e8; border-radius:5px; border:1px solid #c9a84c33; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; font-family:'DM Mono',monospace; color:#1a1a1a; }
.sm-card.red { color:#dc2626; }
.sm-card.new { border:1.5px solid #c9a84c; box-shadow:0 0 6px #c9a84c44; }
.replay-decision { margin-left:auto; text-align:right; }
.replay-dec-label { font-family:'DM Mono',monospace; font-size:13px; font-weight:600; }
.replay-win { font-size:11px; color:#6b7a99; margin-top:2px; font-family:'DM Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANK_NAMES = {2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",
              9:"9",10:"T",11:"J",12:"Q",13:"K",14:"A"}
SUIT_SYMS  = {0:"♠", 1:"♥", 2:"♦", 3:"♣"}
SUIT_RED   = {0:False, 1:True, 2:True, 3:False}
HAND_NAMES = ["High Card","One Pair","Two Pair","Three of a Kind",
              "Straight","Flush","Full House","Four of a Kind","Straight Flush"]
POSITIONS  = ["UTG","UTG+1","UTG+2","MP","MP+1","HJ","CO","BTN","SB","BB"]
FEATURE_COLS = [
    "hand_strength","preflop_equity","pot_odds","spr","position",
    "n_opponents","street","bet_to_call_ratio","high_card","suited",
    "paired","connected","board_paired","board_flush_draw",
    "n_community","stack_ratio",
]

# ── Card helpers ──────────────────────────────────────────────────────────────
def card_id(rank, suit): return (rank - 2) * 4 + suit
def card_rank(card):     return card // 4 + 2
def card_suit(card):     return card % 4
def card_label(card):    return RANK_NAMES[card_rank(card)] + SUIT_SYMS[card_suit(card)]
def is_red(card):        return SUIT_RED[card_suit(card)]

def eval_hand(cards):
    ranks  = sorted([card_rank(c) for c in cards], reverse=True)
    suits  = [card_suit(c) for c in cards]
    rc     = {}
    for r in ranks: rc[r] = rc.get(r, 0) + 1
    counts = sorted(rc.values(), reverse=True)
    n      = len(cards)
    is_flush    = (len(set(suits)) == 1 and n == 5)
    is_straight = (n == 5 and len(set(ranks)) == 5 and max(ranks)-min(ranks) == 4)
    if set(ranks) == {14,2,3,4,5}: is_straight = True
    if is_straight and is_flush:                               return 8
    if counts[0] == 4:                                         return 7
    if len(counts)>1 and counts[0]==3 and counts[1]==2:        return 6
    if is_flush:                                               return 5
    if is_straight:                                            return 4
    if counts[0] == 3:                                         return 3
    if len(counts)>1 and counts[0]==2 and counts[1]==2:        return 2
    if counts[0] == 2:                                         return 1
    return 0

def hand_strength_score(hole, community):
    all_cards = hole + community
    if len(all_cards) < 2: return 0
    return max(eval_hand(list(c))
               for c in combinations(all_cards, min(5, len(all_cards))))

def preflop_equity(hole):
    r1, r2 = sorted([card_rank(c) for c in hole], reverse=True)
    suited = card_suit(hole[0]) == card_suit(hole[1])
    paired = (r1 == r2)
    base   = (r1 + r2) / 28.0
    if paired:               base += 0.15 + (r1-2)*0.01
    if suited:               base += 0.04
    if r1-r2<=2 and not paired: base += 0.02
    return min(base, 1.0)

def suggest_raise_size(pot, stack, bet_to_call, street, hs, bb_size):
    """
    Return (raise_to_amount, sizing_label, reasoning) based on street + hand strength.
    Uses standard GTO sizing conventions (pot fractions).
    """
    total_pot = pot + bet_to_call * 2   # pot after calling
    street_name = ["Pre-flop","Flop","Turn","River"][street]

    # Pick sizing multiplier based on street + hand strength
    if street == 0:  # preflop: use BB multiples
        bb = bb_size if bb_size > 0 else 2
        if hs >= 6:      mult, label = 4.0, "4x open (premium)"
        elif hs >= 3:    mult, label = 3.0, "3x open (standard)"
        else:            mult, label = 2.5, "2.5x open (wide range)"
        raise_to = mult * bb
        reason = f"Standard {label} raise from {['UTG','UTG+1','MP','CO','BTN','BB'][0]}"
    else:
        # Postflop: size as fraction of pot
        if hs >= 6:      frac, label = 1.0,  "Pot-sized (value)"
        elif hs >= 4:    frac, label = 0.75, "¾ pot (strong value)"
        elif hs >= 2:    frac, label = 0.5,  "½ pot (thin value / protection)"
        else:            frac, label = 0.33, "⅓ pot (bluff / probe)"
        raise_to = total_pot * frac
        reason = f"{label} on the {street_name}"

    raise_to = min(raise_to, stack)   # can't raise more than stack
    return raise_to, label, reason

def monte_carlo_equity(hole, community, n_opponents, n_simulations=2000):
    """
    Estimate win/tie/lose probabilities via Monte Carlo simulation.
    Deals random opponent hands and remaining community cards,
    evaluates all hands, returns (win%, tie%, lose%).
    """
    wins = ties = losses = 0
    used = set(hole + community)
    deck_remaining = [c for c in range(52) if c not in used]
    cards_needed   = 5 - len(community)   # community cards still to come

    for _ in range(n_simulations):
        deck = deck_remaining.copy()
        np.random.shuffle(deck)

        idx = 0
        # Deal remaining community cards
        board = community + deck[idx: idx + cards_needed]
        idx  += cards_needed

        # Deal opponent hole cards
        opponent_hands = []
        valid = True
        for _ in range(n_opponents):
            if idx + 2 > len(deck):
                valid = False
                break
            opponent_hands.append([deck[idx], deck[idx+1]])
            idx += 2
        if not valid:
            continue

        my_score  = hand_strength_score(hole, board)
        opp_best  = max(hand_strength_score(opp, board) for opp in opponent_hands)

        if   my_score > opp_best: wins   += 1
        elif my_score == opp_best: ties   += 1
        else:                      losses += 1

    total = wins + ties + losses
    if total == 0:
        return 0.0, 0.0, 1.0
    return wins/total, ties/total, losses/total

def extract_features(hole, community, pot, stack, position, n_opponents, bet_to_call):
    street     = {0:0, 3:1, 4:2, 5:3}[len(community)]
    hs         = hand_strength_score(hole, community) / 8.0
    eq         = preflop_equity(hole)
    total_pot  = pot + bet_to_call
    pot_odds   = bet_to_call / total_pot if total_pot > 0 else 0
    spr        = min(stack / pot, 20.0) / 20.0 if pot > 0 else 0.5
    # Normalize stack relative to pot (SPR-based) rather than a fixed dollar ceiling
    # This makes the feature stakes-agnostic whether inputs are in $ or BB
    stack_ratio = min(stack / max(pot * 10, 1), 1.0)
    r1, r2     = sorted([card_rank(c) for c in hole], reverse=True)
    com_ranks  = [card_rank(c) for c in community]
    com_suits  = [card_suit(c) for c in community]
    return [[
        hs, eq, pot_odds, spr, position/9.0,
        n_opponents/8.0, street/3.0,
        min(bet_to_call/(stack+1), 1.0),
        r1/14.0,
        int(card_suit(hole[0]) == card_suit(hole[1])),
        int(r1 == r2),
        int(r1-r2 <= 2 and r1 != r2),
        int(len(com_ranks) != len(set(com_ranks))) if com_ranks else 0,
        int(any(com_suits.count(s) >= 3 for s in com_suits)) if len(com_suits)>=3 else 0,
        len(community)/5.0, stack_ratio,
    ]]

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """Load model from pickle if compatible, otherwise train from scratch."""

    # Try loading pickle first
    try:
        if os.path.exists("model/poker_model.pkl"):
            with open("model/poker_model.pkl", "rb") as f:
                return pickle.load(f)
    except Exception:
        pass

    # Train from scratch if pickle is incompatible

    def _cr(c): return c // 4 + 2
    def _cs(c): return c % 4

    def _eval(cards):
        ranks = sorted([_cr(c) for c in cards], reverse=True)
        suits = [_cs(c) for c in cards]
        rc = {}
        for r in ranks: rc[r] = rc.get(r, 0) + 1
        counts = sorted(rc.values(), reverse=True)
        n = len(cards)
        fl = len(set(suits)) == 1 and n == 5
        st_ = n == 5 and len(set(ranks)) == 5 and max(ranks) - min(ranks) == 4
        if set(ranks) == {14,2,3,4,5}: st_ = True
        if st_ and fl: return 8
        if counts[0] == 4: return 7
        if len(counts)>1 and counts[0]==3 and counts[1]==2: return 6
        if fl: return 5
        if st_: return 4
        if counts[0] == 3: return 3
        if len(counts)>1 and counts[0]==2 and counts[1]==2: return 2
        if counts[0] == 2: return 1
        return 0

    def _hs(hole, com):
        ac = hole + com
        if len(ac) < 2: return 0
        return max(_eval(list(c)) for c in combinations(ac, min(5, len(ac))))

    def _eq(hole):
        r1,r2 = sorted([_cr(c) for c in hole], reverse=True)
        s = _cs(hole[0])==_cs(hole[1]); p = r1==r2
        b = (r1+r2)/28.0
        if p: b += 0.15+(r1-2)*0.01
        if s: b += 0.04
        if r1-r2<=2 and not p: b += 0.02
        return min(b, 1.0)

    def _feat(hole, com, pot, stack, pos, n_opp, street, btc):
        hs=_hs(hole,com)/8.0; eq=_eq(hole)
        tp=pot+btc; po=btc/tp if tp>0 else 0
        spr=min(stack/pot,20.0) if pot>0 else 10.0
        r1,r2=sorted([_cr(c) for c in hole],reverse=True)
        cs_=[_cs(c) for c in com]; cr_=[_cr(c) for c in com]
        return [hs,eq,po,spr/20.0,pos/9.0,n_opp/8.0,street/3.0,
                min(btc/(stack+1),1.0),r1/14.0,
                int(_cs(hole[0])==_cs(hole[1])),int(r1==r2),
                int(r1-r2<=2 and r1!=r2),
                int(len(cr_)!=len(set(cr_))) if cr_ else 0,
                int(any(cs_.count(s)>=3 for s in cs_)) if len(cs_)>=3 else 0,
                len(com)/5.0, min(stack/max(pot*10,1),1.0)]

    def _label(f, noise=0.08):
        eff=f[1] if f[14]==0 else 0.3*f[1]+0.7*f[0]
        eff+=np.random.normal(0,noise)
        rt=0.58-f[4]*0.06-f[3]*0.04; ct=f[2]+0.03-f[4]*0.03
        if f[7]==0: return 2 if eff>=rt else 1
        if eff>=rt: return 2
        if eff>=ct and f[7]<0.5: return 1
        if (f[11] or f[13]) and f[4]>0.5 and np.random.random()<0.15: return 2
        if np.random.random()<0.05 and f[7]<0.15: return 2
        return 0

    np.random.seed(42)
    rows=[]
    for _ in range(50000):
        deck=list(range(52)); np.random.shuffle(deck)
        hole=[deck[0],deck[1]]
        street=np.random.choice([0,1,2,3],p=[0.35,0.35,0.15,0.15])
        com=deck[2:2+{0:0,1:3,2:4,3:5}[street]]
        pot=np.random.choice([10,20,40,80,150,300],p=[0.2,0.25,0.25,0.15,0.1,0.05])
        stack=np.random.randint(50,2000); pos=np.random.randint(0,10)
        n_opp=np.random.randint(1,9)
        btc=min(np.random.choice([0,5,10,20,50,100],p=[0.2,0.2,0.25,0.2,0.1,0.05]),stack)
        f=_feat(hole,com,pot,stack,pos,n_opp,street,btc)
        rows.append(f+[_label(f)])

    data=np.array(rows); X,y=data[:,:-1],data[:,-1].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    mdl=GradientBoostingClassifier(n_estimators=200,learning_rate=0.08,
                                    max_depth=5,subsample=0.8,random_state=42)
    mdl.fit(Xtr,ytr)
    os.makedirs("model",exist_ok=True)
    with open("model/poker_model.pkl","wb") as f: pickle.dump(mdl,f)
    return mdl

model = load_or_train_model()

# ── Session state ─────────────────────────────────────────────────────────────
if "hole"      not in st.session_state: st.session_state.hole      = []
if "community" not in st.session_state: st.session_state.community = []
if "replay"    not in st.session_state: st.session_state.replay    = []  # list of street snapshots

def toggle_card(cid):
    h, c = st.session_state.hole, st.session_state.community
    if   cid in h: st.session_state.hole      = [x for x in h if x != cid]
    elif cid in c: st.session_state.community  = [x for x in c if x != cid]
    elif len(h) < 2: st.session_state.hole.append(cid)
    elif len(c) < 5: st.session_state.community.append(cid)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# 🃏 Poker Decision AI")
st.markdown("*Pick your hole cards, then community cards. Set the table situation. Get your move.*")
st.divider()

left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-title">Select Cards — first 2 are your hand, next up to 5 are the board</div>', unsafe_allow_html=True)

    hole      = st.session_state.hole
    community = st.session_state.community
    selected  = set(hole + community)

    for suit in range(4):
        cols = st.columns(13)
        for i, rank in enumerate(range(2, 15)):
            cid      = card_id(rank, suit)
            lbl      = RANK_NAMES[rank] + SUIT_SYMS[suit]
            in_hole  = cid in hole
            in_com   = cid in community
            disabled = (cid in selected) and not in_hole and not in_com

            prefix = "✓" if in_hole else ("★" if in_com else "")
            btn_lbl = f"{prefix}\n{lbl}" if prefix else lbl

            if cols[i].button(btn_lbl, key=f"c{cid}",
                              use_container_width=True, disabled=disabled):
                toggle_card(cid)
                st.rerun()

    st.markdown("---")
    col_h, col_c = st.columns(2)

    with col_h:
        st.markdown('<div class="section-title">Your Hand ✓</div>', unsafe_allow_html=True)
        html = '<div class="hand-display">'
        for c in hole:
            html += f'<div class="big-card {"red" if is_red(c) else ""}">{card_label(c)}</div>'
        for _ in range(2 - len(hole)):
            html += '<div class="big-card empty">?</div>'
        st.markdown(html + "</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-title">Board ★</div>', unsafe_allow_html=True)
        html = '<div class="hand-display">'
        for c in community:
            html += f'<div class="big-card {"red" if is_red(c) else ""}">{card_label(c)}</div>'
        for _ in range(5 - len(community)):
            html += '<div class="big-card empty">?</div>'
        st.markdown(html + "</div>", unsafe_allow_html=True)

    if st.button("🗑 Clear All Cards", use_container_width=True):
        st.session_state.hole      = []
        st.session_state.community = []
        st.rerun()

with right:
    st.markdown('<div class="section-title">Table Situation</div>', unsafe_allow_html=True)

    # ── BB / $ toggle ─────────────────────────────────────────────────────────
    input_mode = st.radio("Input mode", ["Big Blinds (BB)", "Dollar Amount ($)"],
                          horizontal=True, label_visibility="collapsed")
    use_bb = input_mode == "Big Blinds (BB)"

    # Always show bb_size first so layout doesn't shift
    bb_size = st.number_input("Big Blind size ($)", 1, 500, 2, step=1)

    if use_bb:
        pot_input         = st.number_input("Pot (BB)",           1.0, 500.0,  6.0, step=0.5)
        stack_input       = st.number_input("Your Stack (BB)",    2.0, 500.0, 100.0, step=5.0)
        bet_to_call_input = st.number_input("Bet to Call (BB)",   0.0, 250.0,   3.0, step=0.5)
        pot         = pot_input         * bb_size
        stack       = stack_input       * bb_size
        bet_to_call = bet_to_call_input * bb_size
        st.caption(f"≈ ${pot:.0f} pot · ${stack:.0f} stack · ${bet_to_call:.0f} to call")
    else:
        pot         = float(st.number_input("Pot Size ($)",    10, 10000, 100, step=10))
        stack       = float(st.number_input("Your Stack ($)",  10, 10000, 500, step=10))
        bet_to_call = float(st.number_input("Bet to Call ($)",  0,  5000,  20, step=5))
        if bb_size > 0:
            st.caption(f"≈ {pot/bb_size:.1f}BB pot · {stack/bb_size:.1f}BB stack · {bet_to_call/bb_size:.1f}BB to call")

    position = st.select_slider("Your Position", options=list(range(10)), value=6,
                                format_func=lambda x: POSITIONS[x])

    # ── Opponent seat picker ──────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:12px">Opponent Positions (click to toggle)</div>', unsafe_allow_html=True)

    if "opp_positions" not in st.session_state:
        st.session_state.opp_positions = set()

    # Build 10 seat buttons in 2 rows of 5
    btn_cols1 = st.columns(5)
    btn_cols2 = st.columns(5)
    all_btn_cols = btn_cols1 + btn_cols2

    for i, pos_name in enumerate(POSITIONS):
        is_your_seat  = (i == position)
        is_selected   = (i in st.session_state.opp_positions)
        label = f"✓ {pos_name}" if is_selected else pos_name
        if all_btn_cols[i].button(
            label,
            key=f"opp_pos_{i}",
            disabled=is_your_seat,
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            if i in st.session_state.opp_positions:
                st.session_state.opp_positions.discard(i)
            else:
                st.session_state.opp_positions.add(i)
            st.rerun()

    n_opponents = len(st.session_state.opp_positions)
    if n_opponents == 0:
        st.caption("⚠️ Select at least one opponent seat")
    else:
        opp_names = ", ".join(POSITIONS[i] for i in sorted(st.session_state.opp_positions))
        st.caption(f"{n_opponents} opponent{'s' if n_opponents > 1 else ''}: {opp_names}")

    st.divider()

    # ── Prediction ────────────────────────────────────────────────────────────
    valid_com = len(community) in {0, 3, 4, 5}
    ready     = len(hole) == 2 and valid_com and model is not None and n_opponents > 0

    if not ready:
        if model is None:
            st.error("Run `python train.py` first to generate the model.")
        elif len(hole) < 2:
            st.info("Select 2 hole cards to get a recommendation.")
        elif not valid_com:
            st.warning("Board must have 0 (pre-flop), 3 (flop), 4 (turn), or 5 (river) cards.")
        elif n_opponents == 0:
            st.info("Select at least one opponent seat above.")
    else:
        X_in  = extract_features(hole, community, pot, stack, position,
                                 n_opponents, bet_to_call)
        probs = model.predict_proba(X_in)[0]
        pred  = int(np.argmax(probs))
        street_idx = {0:0,3:1,4:2,5:3}[len(community)]

        LABELS = ["FOLD",     "CALL",     "RAISE"]
        CSS    = ["fold-box", "call-box", "raise-box"]
        EMOJIS = ["🚫",       "✅",        "🔥"]
        FILLS  = ["fold-fill","call-fill","raise-fill"]
        DEC_COLORS = ["#ef4444", "#22c55e", "#c9a84c"]

        # Decision banner
        st.markdown(f"""
        <div class="decision-box {CSS[pred]}">
          <div style="font-size:2rem">{EMOJIS[pred]}</div>
          <div class="decision-label">{LABELS[pred]}</div>
          <div style="font-size:13px;margin-top:8px;opacity:0.7;font-family:'DM Mono',monospace">
            {probs[pred]:.0%} confidence
          </div>
        </div>""", unsafe_allow_html=True)

        # Probability bars
        for lbl, p, fill in zip(LABELS, probs, FILLS):
            st.markdown(f"""
            <div class="prob-bar-wrap">
              <div class="prob-label"><span>{lbl.title()}</span><span>{p:.1%}</span></div>
              <div class="prob-track"><div class="prob-fill {fill}" style="width:{p*100:.1f}%"></div></div>
            </div>""", unsafe_allow_html=True)

        # Stats row
        hs         = hand_strength_score(hole, community)
        eq         = preflop_equity(hole)
        total_pot  = pot + bet_to_call
        pot_odds   = bet_to_call / total_pot if total_pot > 0 else 0
        eff_eq     = eq if not community else (hs/8.0)*0.7 + eq*0.3

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-box"><div class="stat-val">{HAND_NAMES[hs][:8]}</div><div class="stat-lbl">Best Hand</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-box"><div class="stat-val">{eq:.0%}</div><div class="stat-lbl">Pre-flop Equity</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-box"><div class="stat-val">{pot_odds:.0%}</div><div class="stat-lbl">Pot Odds</div></div>', unsafe_allow_html=True)

        # EV insight
        insights = {
            0: f"Folding is correct — estimated equity ({eff_eq:.0%}) is below the pot odds required ({pot_odds:.0%}).",
            1: f"Calling has positive EV. Your equity ({eff_eq:.0%}) covers the {pot_odds:.0%} pot odds you're getting.",
            2: f"Raise. Strong hand ({HAND_NAMES[hs]}) with {eq:.0%} equity — build the pot while you're ahead.",
        }
        st.markdown(f'<div class="ev-insight">💡 {insights[pred]}</div>', unsafe_allow_html=True)

        # ── Suggested raise size ──────────────────────────────────────────────
        if pred == 2:
            raise_to, sizing_label, reasoning = suggest_raise_size(
                pot, stack, bet_to_call, street_idx, hs, bb_size)
            bb_equiv = f" ({raise_to/bb_size:.1f}BB)" if bb_size > 0 else ""
            st.markdown(f"""
            <div class="raise-size-box">
              <div>
                <div style="font-size:10px;font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:2px;color:#6b7a99;margin-bottom:4px">Suggested Raise To</div>
                <div class="raise-size-amount">${raise_to:.0f}{bb_equiv}</div>
              </div>
              <div class="raise-size-detail">
                <div style="color:#e8dcc8;font-weight:500">{sizing_label}</div>
                <div>{reasoning}</div>
                <div style="margin-top:2px">Into ${pot:.0f} pot · ${stack:.0f} stack remaining</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Win/Loss probability box ──────────────────────────────────────────
        win_p, tie_p, lose_p = monte_carlo_equity(hole, community, n_opponents)

        def bar_width(p): return max(p * 100, 0)
        win_lbl  = f"{win_p:.0%}"  if win_p  >= 0.08 else ""
        tie_lbl  = f"{tie_p:.0%}"  if tie_p  >= 0.05 else ""
        lose_lbl = f"{lose_p:.0%}" if lose_p >= 0.08 else ""

        st.markdown(f"""
        <div class="wl-box">
          <div class="wl-title">Win / Tie / Lose — {n_opponents} opponent{"s" if n_opponents>1 else ""} · 2,000 simulations</div>
          <div class="wl-bar-row">
            <div class="wl-win"  style="width:{bar_width(win_p):.1f}%">{win_lbl}</div>
            <div class="wl-tie"  style="width:{bar_width(tie_p):.1f}%">{tie_lbl}</div>
            <div class="wl-lose" style="width:{bar_width(lose_p):.1f}%">{lose_lbl}</div>
          </div>
          <div class="wl-legend">
            <span><span class="wl-dot" style="background:#22c55e"></span>Win {win_p:.1%}</span>
            <span><span class="wl-dot" style="background:#c9a84c"></span>Tie {tie_p:.1%}</span>
            <span><span class="wl-dot" style="background:#ef4444"></span>Lose {lose_p:.1%}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Conflict warning ──────────────────────────────────────────────────
        # Detect when decision and equity signal disagree meaningfully
        equity_suggests_call  = win_p >= (pot_odds + 0.05)   # hand is +EV to call
        equity_suggests_fold  = win_p < pot_odds              # hand is -EV to call
        model_says_fold       = pred == 0
        model_says_call_raise = pred >= 1

        conflict = (model_says_fold and equity_suggests_call and win_p > 0.45) or \
                   (model_says_call_raise and equity_suggests_fold and pot_odds > 0.25)

        if conflict:
            if model_says_fold and equity_suggests_call:
                msg = (f"The model says <strong>Fold</strong> ({probs[0]:.0%} confidence) "
                       f"but your raw win probability is <strong>{win_p:.0%}</strong> — above the "
                       f"{pot_odds:.0%} pot odds needed to call. The model may be accounting for "
                       f"position or stack pressure. Use your judgment.")
            else:
                msg = (f"The model says <strong>{LABELS[pred]}</strong> but your win probability "
                       f"({win_p:.0%}) is below the pot odds required ({pot_odds:.0%}). "
                       f"The call may not be +EV on raw equity alone.")
            st.markdown(f"""
            <div style="background:#1a1a0a;border:1px solid #d97706;border-radius:8px;
                        padding:12px 16px;margin-top:10px;font-size:12px;color:#fde68a;
                        font-family:'DM Sans',sans-serif;line-height:1.6">
              ⚡ <strong>Signal conflict:</strong> {msg}
            </div>""", unsafe_allow_html=True)

        # ── Save snapshot to replay ───────────────────────────────────────────
        street_names = ["Pre-flop","Flop","Turn","River"]
        snapshot = {
            "street":     street_names[street_idx],
            "community":  list(community),
            "decision":   LABELS[pred],
            "dec_color":  DEC_COLORS[pred],
            "confidence": probs[pred],
            "win_p":      win_p,
            "hand_name":  HAND_NAMES[hs],
        }
        # Only append if this street isn't already recorded
        existing_streets = [s["street"] for s in st.session_state.replay]
        if snapshot["street"] not in existing_streets:
            st.session_state.replay.append(snapshot)

# ── Street-by-Street Replay ───────────────────────────────────────────────────
if st.session_state.replay:
    st.divider()
    street_order = ["Pre-flop","Flop","Turn","River"]
    replay_sorted = sorted(st.session_state.replay,
                           key=lambda s: street_order.index(s["street"]))

    col_title, col_clear = st.columns([4,1])
    with col_title:
        st.markdown("### 🎬 Hand Replay")
    with col_clear:
        if st.button("Clear Replay", use_container_width=True):
            st.session_state.replay = []
            st.rerun()

    rows_html = ""
    for snap in replay_sorted:
        street_com = snap["community"]
        prev_streets = street_order[:street_order.index(snap["street"])]

        # Figure out which cards are "new" this street
        prev_count = {"Pre-flop":0,"Flop":0,"Turn":3,"River":4}[snap["street"]]
        new_from   = prev_count

        # Build small card HTML
        cards_html = ""
        # Hole cards (always shown, never highlighted as new)
        hole_cards = st.session_state.hole
        for c in hole_cards:
            red = "red" if is_red(c) else ""
            cards_html += f'<div class="sm-card {red}">{card_label(c)}</div>'

        cards_html += '<div style="width:6px"></div>'  # separator

        # Community cards
        for i, c in enumerate(street_com):
            red = "red" if is_red(c) else ""
            new = " new" if i >= new_from else ""
            cards_html += f'<div class="sm-card {red}{new}">{card_label(c)}</div>'

        # Empty community slots
        for _ in range(5 - len(street_com)):
            cards_html += '<div class="sm-card" style="background:#0d1f3c;border:1px dashed #2a3a5c;color:#2a3a5c">·</div>'

        rows_html += f"""
        <div class="replay-street-row">
          <div class="replay-street-badge">{snap["street"]}</div>
          <div class="replay-cards">{cards_html}</div>
          <div class="replay-decision">
            <div class="replay-dec-label" style="color:{snap['dec_color']}">{snap['decision']}</div>
            <div class="replay-win">Win {snap['win_p']:.0%} · {snap['hand_name']}</div>
            <div class="replay-win">{snap['confidence']:.0%} confidence</div>
          </div>
        </div>"""

    st.markdown(f'<div class="replay-box">{rows_html}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown("""
**Model**: Gradient Boosting Classifier (scikit-learn) — ~87% accuracy  
**Training data**: 50,000 synthetic hands labeled using GTO (Game Theory Optimal) principles  
**Decision split**: Fold 32% · Call 41% · Raise 28%  
**Features**: Hand strength, pre-flop equity, pot odds, SPR, position, opponents, board texture  
**Target**: Fold / Call / Raise  
**Validation**: 5-fold cross-validation on held-out test set  

> Approximates optimal poker strategy. Real GTO solvers use Nash Equilibrium across millions of hands.
    """)
    if os.path.exists("model/feature_importance.png"):
        st.image("model/feature_importance.png", caption="Feature Importances")
