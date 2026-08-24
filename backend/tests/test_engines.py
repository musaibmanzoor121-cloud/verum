"""
tests/test_engines.py
=====================
Fast unit tests that prove each engine behaves sensibly. Run with:

    cd backend
    pip install -r requirements.txt -r requirements-dev.txt
    pytest -q

These tests are also a great talking point in interviews: they show the
detector actually separates human-like from AI-like inputs, and that the
ethical guardrails (never auto-reject) hold.
"""

from app.schemas import BehaviorPayload, MouseEvent
from app.services import ai_analyzer, behavior_scorer, consistency_checker

# A deliberately "human" sample: uneven sentences, concrete specifics, no filler.
HUMAN_RESUME = (
    "I fixed the checkout bug that lost us $4k a week. Took three days. "
    "At Zappos I rebuilt the returns queue in Go and cut latency from 800ms to 90ms. "
    "Also mentored two interns. One shipped our first GraphQL endpoint. "
    "I like small teams. Less politics, more shipping."
)

# A deliberately "AI/template" sample: uniform sentences, heavy buzzwords.
AI_RESUME = (
    "I am a results-driven and detail-oriented professional. "
    "I leveraged cutting-edge solutions to spearhead cross-functional initiatives. "
    "I am passionate about delivering best-in-class, robust solutions seamlessly. "
    "Furthermore, I utilized innovative solutions to drive strategic initiatives. "
    "Moreover, I leveraged synergy to move the needle across dynamic teams."
)


def test_ai_text_scores_higher_than_human_text():
    human = ai_analyzer.analyze_content(HUMAN_RESUME)
    ai = ai_analyzer.analyze_content(AI_RESUME)
    assert ai.ai_likelihood_score > human.ai_likelihood_score
    assert ai.buzzword_density > human.buzzword_density
    # Evidence must be attached, not just a number (explainability contract).
    assert len(ai.evidence) >= 1


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
