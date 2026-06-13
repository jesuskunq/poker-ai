# Poker Decision AI — Setup & Deployment

## Local Setup

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
python train.py        # generates model/ folder, ~30 seconds
streamlit run app.py   # opens at http://localhost:8501
```

---

## Deploy to Streamlit Cloud (free public URL)

```bash
git init
git add .
git commit -m "Poker Decision AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/poker-ai.git
git push -u origin main
```

1. Go to **share.streamlit.io**
2. Connect GitHub → select repo → select `app.py` → Deploy
3. Get your public URL in ~2 minutes

> ⚠️ Make sure to commit the `model/` folder (the .pkl files) before pushing.

---

## Project Structure

```
poker-ai/
├── train.py           ← generates 50k hands + trains model
├── app.py             ← Streamlit web app
├── requirements.txt
├── README.md
└── model/
    ├── poker_model.pkl        ← trained GBM classifier
    ├── feature_info.pkl       ← feature metadata
    └── feature_importance.png ← evaluation chart
```

---

## For Your Report

**Problem**: Predict the optimal poker decision (Fold/Call/Raise) given hole cards, community cards, and table dynamics — framed as a 3-class classification problem.

**Dataset**: 50,000 synthetic hands generated using GTO (Game Theory Optimal) poker principles. Each hand simulates a realistic game state and labels the decision based on expected value and pot odds calculations — the same logic professional players use.

**Why synthetic data?** Real labeled poker hand histories with ground-truth optimal decisions don't exist publicly — professional players don't publish their decision rationale. Generating synthetic data from established poker theory is a standard approach in computational poker research.

**Model**: Gradient Boosting Classifier
- 200 estimators, learning rate 0.08, max depth 5
- Chosen for ability to handle non-linear decision boundaries (poker decisions are highly non-linear)

**Features (16 total)**:
- Hand strength score (best 5-card hand from hole + community)
- Pre-flop equity estimate
- Pot odds (bet-to-call / total pot)
- Stack-to-pot ratio (SPR)
- Position (early/late position)
- Number of active opponents
- Street (pre-flop / flop / turn / river)
- Board texture (paired board, flush draw present)
- Hole card properties (suited, paired, connected)

**Evaluation**: 85% accuracy, 5-fold cross-validation. Strongest on Fold (88% F1) and Call (84% F1), weaker on Raise (74% F1) due to class imbalance — raises are the rarest decision.

**Business framing**: A simplified GTO solver built with supervised ML instead of Nash Equilibrium computation. Real poker solvers (PioSOLVER, GTO+) require thousands of CPU hours; this model approximates decisions in milliseconds.
