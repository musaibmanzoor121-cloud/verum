"""
tests/test_engines.py
=====================
Fast unit tests that prove each engine behaves sensibly. Run with:

    cd backend
    pip install -r requirements.txt -r requirements-dev.txt
    pytest -q

These tests are also a great talking point in interviews: they show the
detector actually separates human-like from AI-like inputs, that it catches
*modern* (2025+) chat-model writing — not just clumsy old buzzword spam — and,
crucially, that it does NOT falsely flag genuine, specific human writing.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import BehaviorPayload, MouseEvent
from app.services import ai_analyzer, behavior_scorer, consistency_checker

client = TestClient(app)

# A deliberately "human" sample: uneven sentences, concrete specifics, no filler.
HUMAN_RESUME = (
    "I fixed the checkout bug that lost us $4k a week. Took three days. "
    "At Zappos I rebuilt the returns queue in Go and cut latency from 800ms to 90ms. "
    "Also mentored two interns. One shipped our first GraphQL endpoint. "
    "I like small teams. Less politics, more shipping."
)

# A matching human-written cover letter: casual, specific, uneven.
HUMAN_COVER = (
    "Hi Priya,\n\n"
    "Saw the backend opening on your careers page and figured I'd reach out. "
    "I spent the last two years at Zappos fixing the returns system nobody wanted to touch. "
    "It was slow and it dropped orders on busy days. I rewrote the queue in Go and "
    "got p95 latency from 800ms down to 90ms. That one project taught me more about "
    "databases than my degree did. I'm not the flashiest engineer, but I ship, I write "
    "tests, and I stick around when things break at 2am. Happy to walk through the code "
    "if that helps. Thanks for reading.\n\n"
    "Ben"
)

# A deliberately "AI/template" sample: uniform sentences, heavy buzzwords (old style).
AI_RESUME = (
    "I am a results-driven and detail-oriented professional. "
    "I leveraged cutting-edge solutions to spearhead cross-functional initiatives. "
    "I am passionate about delivering best-in-class, robust solutions seamlessly. "
    "Furthermore, I utilized innovative solutions to drive strategic initiatives. "
    "Moreover, I leveraged synergy to move the needle across dynamic teams."
)

# A realistic MODERN chat-model sample (the hard case): varied sentence length,
# few clumsy buzzwords, but full of AI "cadence" phrasing, em-dashes and triads.
MODERN_AI_RESUME = (
    "Machine Learning Engineer with a strong foundation in building scalable, "
    "production-ready systems. Proven ability to design, develop, and deploy "
    "end-to-end machine learning pipelines. Passionate about leveraging data to "
    "drive meaningful impact and deliver reliable, innovative solutions."
)

MODERN_AI_COVER = (
    "Dear Hiring Manager,\n\n"
    "I am writing to express my enthusiasm for the Machine Learning Engineer role "
    "at your organization. As a passionate and dedicated professional, I have always "
    "been drawn to the intersection of data, technology, and real-world impact. "
    "Throughout my career, I have honed my ability to design, build, and deploy "
    "scalable machine learning systems that deliver meaningful results. In my previous "
    "role, I spearheaded initiatives that not only improved model accuracy but also "
    "streamlined our deployment pipeline — a testament to my commitment to both "
    "innovation and reliability. I thrive in fast-paced environments where "
    "collaboration, creativity, and continuous learning are valued. I am confident "
    "that my unique blend of technical expertise and passion for solving complex "
    "problems would make me a valuable addition to your team. Thank you for considering "
    "my application. I look forward to the opportunity to contribute to your "
    "organization's continued success.\n\n"
    "Sincerely,\nAlex Morgan"
)


# ---------------------------------------------------------------------------
# Engine A — content authenticity
# ---------------------------------------------------------------------------
def test_ai_text_scores_higher_than_human_text():
    human = ai_analyzer.analyze_content(HUMAN_RESUME)
    ai = ai_analyzer.analyze_content(AI_RESUME)
    assert ai.ai_likelihood_score > human.ai_likelihood_score
    assert ai.buzzword_density > human.buzzword_density
    # Evidence must be attached, not just a number (explainability contract).
    assert len(ai.evidence) >= 1


def test_modern_ai_writing_is_detected():
    """The hard case: polished 2025-era chat-model prose (varied sentences,
    subtle vocabulary). It must still land clearly above the human sample and
    reach at least the 'needs review' range."""
    combined = f"{MODERN_AI_RESUME}\n\n{MODERN_AI_COVER}"
    result = ai_analyzer.analyze_content(combined)
    # Should be a substantial score — modern AI is the whole point of the upgrade.
    assert result.ai_likelihood_score >= 0.45
    # The new signals should actually fire and be explained.
    signals = {e.signal for e in result.evidence}
    assert "ai_cadence_phrases" in signals
    assert {"em_dash_usage", "triadic_lists"} & signals


def test_genuine_human_writing_is_not_false_flagged():
    """Fairness guardrail: a real, specific, casually-written application must
    NOT be flagged. We would rather miss AI than wrongly accuse a person."""
    combined = f"{HUMAN_RESUME}\n\n{HUMAN_COVER}"
    result = ai_analyzer.analyze_content(combined)
    assert result.ai_likelihood_score < 0.34


def test_modern_ai_beats_human_by_a_clear_margin():
    ai = ai_analyzer.analyze_content(f"{MODERN_AI_RESUME}\n\n{MODERN_AI_COVER}")
    human = ai_analyzer.analyze_content(f"{HUMAN_RESUME}\n\n{HUMAN_COVER}")
    assert ai.ai_likelihood_score - human.ai_likelihood_score >= 0.30


# ---------------------------------------------------------------------------
# Engine B — behavior / bot
# ---------------------------------------------------------------------------
def test_honeypot_is_a_strong_bot_signal():
    payload = BehaviorPayload(
        form_load_time=0, submit_time=10_000, honeypot="i-am-a-bot"
    )
    result = behavior_scorer.score_behavior(payload, user_agent="Mozilla/5.0")
    assert result.honeypot_triggered is True
    assert result.bot_likelihood_score > 0.9


def test_fast_submit_is_flagged():
    payload = BehaviorPayload(form_load_time=0, submit_time=500)  # 0.5s
    result = behavior_scorer.score_behavior(payload, user_agent="Mozilla/5.0")
    assert result.time_to_submit_seconds is not None
    assert result.bot_likelihood_score > 0.3


def test_human_like_mouse_and_keys_are_not_flagged_as_bot():
    # Varied mouse path + varied typing rhythm => low bot score.
    mouse = [MouseEvent(t=i * 20, x=i * 3 + (i % 5), y=i * 2 + (i % 7)) for i in range(30)]
    keys = [90, 140, 70, 210, 55, 180, 120, 60, 250, 95, 130]
    payload = BehaviorPayload(
        form_load_time=0, submit_time=12_000, mouse_events=mouse, key_intervals_ms=keys
    )
    result = behavior_scorer.score_behavior(
        payload, user_agent="Mozilla/5.0 (Windows NT 10.0) Chrome/125"
    )
    assert result.bot_likelihood_score < 0.5


# ---------------------------------------------------------------------------
# Cross-document consistency
# ---------------------------------------------------------------------------
def test_consistency_flags_cover_letter_only_claims():
    resume = "Software engineer skilled in Python and Django. Worked at Acme Corp."
    cover = "At Google Cloud I used TensorFlow and Kubernetes to build ML systems."
    result = consistency_checker.compare_documents(resume, cover)
    assert result.checked is True
    assert len(result.mismatches) >= 1  # TensorFlow/Kubernetes/Google not in resume


def test_no_cover_letter_means_unchecked():
    result = consistency_checker.compare_documents("resume text", "")
    assert result.checked is False
    assert result.consistency_score == 1.0


def test_consistency_ignores_document_scaffolding():
    """Regression test for the junk-evidence bug: greetings, titles and job
    titles must NEVER be reported as 'entities missing from the resume'."""
    resume = "Experienced developer. Built systems in Python. Studied at MIT."
    cover = (
        "COVER LETTER\n"
        "Dear Hiring Manager,\n"
        "I am applying for the ML Engineer role. Built systems in Python.\n"
        "Best Regards,\n"
        "Alex"
    )
    result = consistency_checker.compare_documents(resume, cover)
    junk = ["cover letter", "dear hiring manager", "ml engineer", "best regards", "hiring manager"]
    for m in result.mismatches:
        low = m.lower()
        assert not any(j in low for j in junk), f"scaffolding leaked into evidence: {m}"


# ---------------------------------------------------------------------------
# End-to-end fusion (the exact real-world scenario: a HUMAN pastes AI text)
# ---------------------------------------------------------------------------
def test_endtoend_human_submitting_ai_text_is_not_marked_authentic():
    """A real person (human behavior) submitting ChatGPT-written text should NOT
    come back 'Likely authentic' — the AI content must drive it to review."""
    mouse = [MouseEvent(t=i * 20, x=i * 3 + (i % 5), y=i * 2 + (i % 7)) for i in range(30)]
    keys = [90, 140, 70, 210, 55, 180, 120, 60, 250, 95, 130]
    resp = client.post("/api/analyze-text", json={
        "resume_text": MODERN_AI_RESUME,
        "cover_letter_text": MODERN_AI_COVER,
        "behavior": {
            "form_load_time": 0, "submit_time": 45_000,  # 45s: clearly human timing
            "mouse_events": [e.model_dump() for e in mouse],
            "key_intervals_ms": keys,
        },
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] != "Likely authentic"


def test_endtoend_genuine_human_stays_authentic():
    """The fairness counterpart: a genuine human application stays 'Likely
    authentic' end-to-end (no false alarm from the stronger content engine)."""
    mouse = [MouseEvent(t=i * 20, x=i * 3 + (i % 5), y=i * 2 + (i % 7)) for i in range(30)]
    keys = [90, 140, 70, 210, 55, 180, 120, 60, 250, 95, 130]
    resp = client.post("/api/analyze-text", json={
        "resume_text": HUMAN_RESUME,
        "cover_letter_text": HUMAN_COVER,
        "behavior": {
            "form_load_time": 0, "submit_time": 40_000,
            "mouse_events": [e.model_dump() for e in mouse],
            "key_intervals_ms": keys,
        },
    })
    assert resp.status_code == 200
    assert resp.json()["verdict"] == "Likely authentic"
