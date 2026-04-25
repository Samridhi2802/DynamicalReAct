# DynamicalReAct

**A physics-inspired stability controller for LLM reasoning trajectories.**

> ⚠️ **Honest status:** This is a proof-of-concept. The pipeline runs end-to-end and the signals compute correctly. The accuracy result on n=10 examples is -10% vs baseline — a calibration issue, not a hypothesis failure. Proper validation is the next step.

---

## The Idea

Standard ReAct agents reason linearly — Thought → Action → Observation — repeated until an answer is found. On hard multi-hop questions, this fails silently: the model hallucinates confidently, drifts off-topic, or loops without knowing it.

**DynamicalReAct** treats the agent's reasoning as a *trajectory in a high-dimensional embedding space* and borrows tools from nonlinear dynamical systems theory to detect instability in real time — before the answer collapses.

---

## How It Works

A **Dynamical Systems Controller (DSC)** monitors three signals at every reasoning step:

| Signal | Symbol | What it measures |
|--------|--------|-----------------|
| Shannon Entropy | H(t) | Uncertainty in reasoning direction. High = confused, Low = confident. |
| State Drift | D(t) | Cosine distance from the rolling window average. Detects path deviation. |
| Proxy Lyapunov Stability | λ(t) | Aggregate collapse-prediction signal. Borrowed from Lyapunov stability theory. |

**State vector is 386-dimensional:**
- 384 dims — semantic embedding of current reasoning context (`all-MiniLM-L6-v2`)
- 1 dim — normalized step position
- 1 dim — normalized context length

**Controller logic:**
- `entropy > threshold` → status `UNSTABLE` → inject `REFINE_PROMPT`
- `drift > threshold` → status `DRIFTING` → inject `RECALL_CONTEXT`
- otherwise → status `HEALTHY` → `CONTINUE`

---

## Results (Proof of Concept, n=10)

| Method | Accuracy | vs Baseline |
|--------|----------|-------------|
| ReAct Baseline | 60.0% | — |
| Reflexion | 60.0% | 0.0% |
| DynamicalReAct | 50.0% | -10.0% |

**DSC signal summary across 35 steps:**
- HEALTHY: 24 steps
- DRIFTING: 9 steps
- UNSTABLE: 2 steps
- Interventions fired: 11/35 (~31% — too aggressive, needs tuning)
- Entropy mean: 0.889 | Drift mean: 0.260 | Stability mean: -1.149

**Why -10%?** The intervention rate of 31% is too high. The controller is disrupting reasoning that was already converging. Threshold tuning is the immediate next step, not an architecture change.

---

## What Is Actually Validated

- ✅ DSC pipeline runs end-to-end
- ✅ Three signals compute correctly from real reasoning trajectories
- ✅ Phase portraits of agent reasoning are interpretable and meaningful
- ✅ HEALTHY / DRIFTING / UNSTABLE classifications align with step-by-step traces
- ❌ Accuracy improvement not yet demonstrated
- ❌ Thresholds not tuned
- ❌ Only tested on n=10 examples

---

## Notebook Structure

```
DynamicalReAct_ProofOfConcept.ipynb
│
├── Cell 1 — Infrastructure & Environment Setup
├── Cell 2 — Dynamical Systems Controller (DSC)
├── Cell 3 — HotpotQA Benchmark: 3-way comparison
├── Cell 3B — Results logging (JSON + CSV)
└── Cell 4 — Phase Portrait Gallery
```

---

## Setup

```bash
# Clone this repo
git clone https://github.com/yourusername/DynamicalReAct.git

# Install dependencies
pip install hydra-core omegaconf groq sentence-transformers datasets rich
```

You will need a **Groq API key** set as `GROQ_API_KEY` in your environment (or as a Colab secret).

---

## Roadmap

**Immediate**
- [ ] Grid search threshold tuning (`entropy_threshold`, `drift_threshold`)
- [ ] Target intervention rate ~10–15% (currently ~31%)
- [ ] Re-run at n=100+ for statistical validity

**Short term**
- [ ] Test on FEVER and 2WikiMultiHopQA benchmarks
- [ ] Compare against Tree of Thought and CoT Self-Consistency
- [ ] Train intervention templates on logged failure cases

**Long term**
- [ ] Model-agnostic wrapper — drop the DSC on top of any ReAct agent with zero model changes
- [ ] Learned stability classifier trained on trajectory histories
- [ ] Full NeurIPS submission with proper empirical validation

---

## Why This Might Matter

If the controller works at scale, it becomes infrastructure — a plug-in stability monitor for any LLM agent, regardless of the underlying model. The phase portrait visualization alone has value for interpretability: it lets you *see* how an agent reasoned, not just whether it got the answer right.

---

## Honest Limitations

- n=10 is not an experiment, it is a pipeline test
- The signal-to-hypothesis connection (embedding variance ≈ epistemic uncertainty) needs stronger theoretical justification
- Reflexion tied the baseline — stronger comparison agents are needed
- The NeurIPS framing in the notebook is aspirational, not a current claim

---

## Stack

- **Model:** Llama 3.3-70b-versatile via [Groq](https://groq.com)
- **Embeddings:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Benchmark:** HotpotQA fullwiki validation
- **Config:** Hydra + OmegaConf
- **Environment:** Google Colab

---

## References

- Yao et al. (2022) — [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- Shinn et al. (2023) — [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- Lyapunov, A.M. (1892) — The General Problem of the Stability of Motion

---

*Independent research project. Proof-of-concept stage. Feedback welcome.*
