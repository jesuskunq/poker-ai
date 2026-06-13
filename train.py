"""
Poker Decision Predictor — Training Script
Generates synthetic hand histories based on GTO poker principles,
trains a Gradient Boosting classifier to predict Fold / Call / Raise.
Run: python train.py
Output: model/poker_model.pkl, model/feature_info.pkl
"""

import numpy as np
import pandas as pd
import pickle
import os
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Card utilities ────────────────────────────────────────────────────────────
def card_rank(card): return card // 4 + 2   # 2–14
def card_suit(card): return card % 4
def make_deck():     return list(range(52))

def eval_hand(cards):
    """Return hand category 0–8 for a 2–5 card combo."""
    ranks  = sorted([card_rank(c) for c in cards], reverse=True)
    suits  = [card_suit(c) for c in cards]
    rc     = {}
    for r in ranks:
        rc[r] = rc.get(r, 0) + 1
    counts = sorted(rc.values(), reverse=True)
    n      = len(cards)

    is_flush    = (len(set(suits)) == 1 and n == 5)
    is_straight = (n == 5 and len(set(ranks)) == 5 and
                   max(ranks) - min(ranks) == 4)
    # Wheel straight A-2-3-4-5
    if set(ranks) == {14, 2, 3, 4, 5}:
        is_straight = True

    if is_straight and is_flush:                          return 8
    if counts[0] == 4:                                    return 7
    if len(counts) > 1 and counts[0] == 3 and counts[1] == 2: return 6
    if is_flush:                                          return 5
    if is_straight:                                       return 4
    if counts[0] == 3:                                    return 3
    if len(counts) > 1 and counts[0] == 2 and counts[1] == 2: return 2
    if counts[0] == 2:                                    return 1
    return 0

def hand_strength(hole, community):
    """Best 5-card hand score (0–8) from hole + community cards."""
    all_cards = hole + community
    if len(all_cards) < 2:
        return 0
    return max(eval_hand(list(combo))
               for combo in combinations(all_cards, min(5, len(all_cards))))

def preflop_equity(hole):
    """Estimate pre-flop equity (0–1) from hole cards."""
    r1, r2  = sorted([card_rank(c) for c in hole], reverse=True)
    suited  = card_suit(hole[0]) == card_suit(hole[1])
    paired  = (r1 == r2)
    base    = (r1 + r2) / 28.0
    if paired:               base += 0.15 + (r1 - 2) * 0.01
    if suited:               base += 0.04
    if r1 - r2 <= 2 and not paired: base += 0.02
    return min(base, 1.0)

# ── Feature extraction ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "hand_strength", "preflop_equity", "pot_odds", "spr", "position",
    "n_opponents", "street", "bet_to_call_ratio", "high_card", "suited",
    "paired", "connected", "board_paired", "board_flush_draw",
    "n_community", "stack_ratio",
]

def extract_features(hole, community, pot, stack, position,
                     n_opponents, street, bet_to_call):
    """Return feature dict for one hand situation."""
    hs         = hand_strength(hole, community)
    eq         = preflop_equity(hole)
    total_pot  = pot + bet_to_call
    pot_odds   = bet_to_call / total_pot if total_pot > 0 else 0
    spr        = min(stack / pot, 20.0) if pot > 0 else 10.0
    r1, r2     = sorted([card_rank(c) for c in hole], reverse=True)
    suited     = int(card_suit(hole[0]) == card_suit(hole[1]))
    paired     = int(r1 == r2)
    connected  = int(r1 - r2 <= 2 and not paired)
    com_suits  = [card_suit(c) for c in community]
    com_ranks  = [card_rank(c) for c in community]
    board_paired      = int(len(com_ranks) != len(set(com_ranks))) if com_ranks else 0
    board_flush_draw  = (int(any(com_suits.count(s) >= 3 for s in com_suits))
                         if len(com_suits) >= 3 else 0)

    return {
        "hand_strength":    hs / 8.0,
        "preflop_equity":   eq,
        "pot_odds":         pot_odds,
        "spr":              spr / 20.0,
        "position":         position / 5.0,
        "n_opponents":      n_opponents / 8.0,
        "street":           street / 3.0,
        "bet_to_call_ratio": min(bet_to_call / (stack + 1), 1.0),
        "high_card":        r1 / 14.0,
        "suited":           suited,
        "paired":           paired,
        "connected":        connected,
        "board_paired":     board_paired,
        "board_flush_draw": board_flush_draw,
        "n_community":      len(community) / 5.0,
        "stack_ratio":      min(stack / 1000.0, 1.0),
    }

# ── GTO decision labeling ─────────────────────────────────────────────────────
def gto_decision(f, noise=0.08):
    """
    Approximate GTO decision: 0=Fold, 1=Call, 2=Raise.
    Key fix: call threshold now tightly tracks pot odds (equity > pot_odds to call),
    and raise threshold is lowered so the model isn't overly tight.
    """
    eff_eq = (f["preflop_equity"] if f["n_community"] == 0
              else 0.3 * f["preflop_equity"] + 0.7 * f["hand_strength"])
    eff_eq += np.random.normal(0, noise)

    # Raise: strong hand relative to board + position advantage
    raise_thresh = 0.58 - f["position"] * 0.06 - f["spr"] * 0.04

    # Call: equity just needs to beat pot odds (fundamental theorem of poker)
    # Small margin of 0.03 instead of 0.10 — much less tight
    call_thresh = f["pot_odds"] + 0.03 - f["position"] * 0.03

    # Free to see if no bet (check/limp) — always at least call
    if f["bet_to_call_ratio"] == 0:
        return 2 if eff_eq >= raise_thresh else 1

    if eff_eq >= raise_thresh:
        return 2
    if eff_eq >= call_thresh and f["bet_to_call_ratio"] < 0.5:
        return 1
    # Semi-bluff raises: draws + position
    if (f["connected"] or f["board_flush_draw"]) and f["position"] > 0.5 and np.random.random() < 0.15:
        return 2
    # Pure bluff raise occasionally
    if np.random.random() < 0.05 and f["bet_to_call_ratio"] < 0.15:
        return 2
    return 0

# ── Dataset generation ────────────────────────────────────────────────────────
print("Generating poker hand dataset...")
np.random.seed(42)
N    = 50_000
rows = []

for _ in range(N):
    deck = make_deck()
    np.random.shuffle(deck)

    hole    = [deck[0], deck[1]]
    street  = np.random.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.15, 0.15])
    n_com   = {0: 0, 1: 3, 2: 4, 3: 5}[street]
    community = deck[2:2 + n_com]

    pot          = np.random.choice([10, 20, 40, 80, 150, 300],
                                    p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])
    stack        = np.random.randint(50, 2000)
    position     = np.random.randint(0, 6)
    n_opponents  = np.random.randint(1, 9)
    bet_to_call  = min(
        np.random.choice([0, 5, 10, 20, 50, 100], p=[0.2, 0.2, 0.25, 0.2, 0.1, 0.05]),
        stack,
    )

    f = extract_features(hole, community, pot, stack, position,
                         n_opponents, street, bet_to_call)
    rows.append({**f, "decision": gto_decision(f)})

df = pd.DataFrame(rows)
print(f"  {len(df):,} hands | "
      f"Fold={( df.decision==0).mean():.1%}  "
      f"Call={(df.decision==1).mean():.1%}  "
      f"Raise={(df.decision==2).mean():.1%}")

# ── Train ─────────────────────────────────────────────────────────────────────
X = df[FEATURE_COLS]
y = df["decision"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08,
    max_depth=5, subsample=0.8, random_state=42,
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
cv     = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n── Evaluation ──────────────────────────────────────")
print(f"  CV Accuracy:   {cv.mean():.4f} ± {cv.std():.4f}")
print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Fold", "Call", "Raise"]))

# ── Charts ────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
imp.head(10).plot(kind="bar", ax=ax, color="#c9a84c")
ax.set_title("Top 10 Feature Importances — Poker Decision Model")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("model/feature_importance.png", dpi=150)
print("  Saved model/feature_importance.png")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("model/poker_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/feature_info.pkl", "wb") as f:
    pickle.dump({"feature_cols": FEATURE_COLS}, f)

print("\n✓ model/poker_model.pkl")
print("✓ model/feature_info.pkl")
print("\nDone. Run: streamlit run app.py")
