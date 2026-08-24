"""
services/consistency_checker.py
===============================
NEW cross-document engine — "Do the resume and cover letter tell the SAME story?"

Real applicants reuse the same facts across documents: the same employers,
schools, and skills. A cover letter generated separately from a resume (e.g.
"write me a cover letter for a data science job") often invents companies or
skills that never appear in the resume — a subtle but useful authenticity
signal that single-document detectors completely miss.

Approach (deliberately simple & explainable):
  * Pull "key claims" from the cover letter: named skills + multi-word
    proper nouns (likely employers / universities / products).
  * Check whether each claim is actually supported by the resume text.
  * Any cover-letter claim missing from the resume becomes a flagged mismatch,
    WITH the specific term shown as evidence.

We keep this intentionally lightweight (no external NLP model needed).

Care taken here: cover letters are full of *document scaffolding* — greetings
("Dear Hiring Manager"), titles ("COVER LETTER"), section headers, job titles
("ML Engineer"), sign-offs ("Best Regards"). None of that is an "employer the
resume forgot to mention", so we must NOT flag it. Getting this wrong produces
embarrassing junk evidence, so entity extraction below is deliberately strict.
"""

from __future__ import annotations

import re
from typing import List, Set

from app.schemas import ConsistencyAnalysis, EvidenceItem

# A compact skill vocabulary. Extend freely — it only affects recall.
SKILL_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "react", "angular", "vue", "node", "django", "flask",
    "fastapi", "spring", "tensorflow", "pytorch", "scikit-learn", "keras",
    "pandas", "numpy", "docker", "kubernetes", "aws", "azure", "gcp",
    "machine learning", "deep learning", "nlp", "computer vision", "llm",
    "data science", "data analysis", "tableau", "power bi", "excel", "git",
    "linux", "spark", "hadoop", "kafka", "redis", "postgresql", "mongodb",
    "html", "css", "tailwind", "graphql", "rest", "microservices",
}

# Generic words we strip from the EDGES of a candidate entity, and that on
# their own never make something a real "employer/school". Prepositions,
# articles, sentence-openers, months, days, etc.
EDGE_STOPWORDS = {
    "i", "the", "my", "a", "an", "we", "this", "that", "as", "in", "with",
    "for", "at", "on", "of", "to", "and", "or", "but", "so", "then", "also",
    "during", "after", "before", "throughout", "currently", "previously",
    "recently", "additionally", "moreover", "furthermore", "however",
    "therefore", "here", "there", "while", "when", "where", "am", "is", "are",
    "was", "were", "have", "has", "had", "you", "your", "their", "his", "her",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

# If ANY word of a candidate phrase is one of these, the whole phrase is
# document scaffolding — a greeting, a section header, a job title, or a
# sign-off — NOT an employer the resume failed to mention. Drop it entirely.
HARD_SCAFFOLDING = {
    # document types / section headers
    "cover", "letter", "resume", "cv", "curriculum", "vitae", "objective",
    "summary", "profile", "experience", "education", "skills", "projects",
    "project", "certifications", "certification", "references", "reference",
    "contact", "portfolio", "achievements", "interests", "declaration",
    # salutations & sign-offs
    "dear", "hello", "hi", "greetings", "sincerely", "regards", "best",
    "warm", "warmest", "kind", "kindly", "yours", "faithfully", "respectfully",
    "cheers", "thanks", "thank", "sir", "madam", "whom", "concern",
    # job-title / role words (a title is not an employer)
    "hiring", "manager", "engineer", "developer", "scientist", "analyst",
    "intern", "internship", "associate", "senior", "junior", "lead", "leader",
    "specialist", "consultant", "coordinator", "assistant", "director",
    "officer", "administrator", "architect", "designer", "recruiter",
    "applicant", "candidate", "position", "role", "application", "team",
    "department", "company", "organization", "organisation",
}


def _skills_in(text: str) -> Set[str]:
    low = text.lower()
    return {s for s in SKILL_KEYWORDS if s in low}


def _clean_candidate(phrase: str) -> str:
    """Strip leading/trailing filler words from a captured Capitalized phrase.

    'At Google Cloud' -> 'Google Cloud'   (leading preposition removed)
    'The New York'    -> 'New York'
    """
    words = phrase.split()
    while words and words[0].lower() in EDGE_STOPWORDS:
        words.pop(0)
    while words and words[-1].lower() in EDGE_STOPWORDS:
        words.pop()
    return " ".join(words)


def _proper_nouns_in(text: str) -> Set[str]:
    """Multi-word Capitalized phrases that look like real orgs/schools/products.

    Strict on purpose (see module docstring). We process the text LINE BY LINE
    so a title line and the greeting line below it can never merge into one
    bogus phrase like 'COVER LETTER Dear Hiring Manager'.
    """
    found: Set[str] = set()
    for line in text.splitlines():
        for m in re.finditer(r"\b([A-Z][A-Za-z&.]+(?:\s+[A-Z][A-Za-z&.]+)+)\b", line):
            phrase = _clean_candidate(m.group(1).strip())
            words = phrase.split()
            # Need at least two words left after trimming edges.
            if len(words) < 2:
                continue
            lowered = [w.lower() for w in words]
            # Drop anything containing document scaffolding / a job title.
            if any(w in HARD_SCAFFOLDING for w in lowered):
                continue
            # Drop if what's left is nothing but generic edge words.
            if all(w in EDGE_STOPWORDS for w in lowered):
                continue
            if len(phrase) >= 5:
                found.add(phrase)
    return found


def compare_documents(resume_text: str, cover_letter_text: str) -> ConsistencyAnalysis:
    """Return a ConsistencyAnalysis; if there is no cover letter, mark unchecked."""
    if not cover_letter_text or not cover_letter_text.strip():
        return ConsistencyAnalysis(
            consistency_score=1.0, checked=False, mismatches=[], evidence=[]
        )

    resume_low = resume_text.lower()
    evidence: List[EvidenceItem] = []
    mismatches: List[str] = []

    # --- Skills claimed in the cover letter but absent from the resume ----
    cl_skills = _skills_in(cover_letter_text)
    for skill in sorted(cl_skills):
        if skill not in resume_low:
            mismatches.append(f"Skill '{skill}' is highlighted in the cover letter but not present in the resume.")
            evidence.append(EvidenceItem(
                engine="consistency", signal="skill_mismatch",
                detail=f"The cover letter emphasizes '{skill}', but the resume never mentions it.",
                severity="medium", excerpt=skill,
            ))

    # --- Employers / schools named in the cover letter, missing in resume -
    cl_orgs = _proper_nouns_in(cover_letter_text)
    for org in sorted(cl_orgs):
        if org.lower() not in resume_low:
            mismatches.append(f"'{org}' is referenced in the cover letter but does not appear in the resume.")
            evidence.append(EvidenceItem(
                engine="consistency", signal="entity_mismatch",
                detail=f"The cover letter references '{org}', which is absent from the resume.",
                severity="low", excerpt=org,
            ))

    # --- Score: fraction of cover-letter claims supported by the resume ---
    total_claims = len(cl_skills) + len(cl_orgs)
    supported = total_claims - len(mismatches)
    score = 1.0 if total_claims == 0 else max(0.0, supported / total_claims)

    return ConsistencyAnalysis(
        consistency_score=round(score, 3),
        checked=True,
        mismatches=mismatches,
        evidence=evidence,
    )
