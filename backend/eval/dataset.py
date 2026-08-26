"""
eval/dataset.py
===============
A small, HAND-LABELED starter dataset for measuring Engine A (content
authenticity). Each item is a realistic resume + cover-letter snippet labelled
as written by a HUMAN or by an AI assistant.

Why this matters
----------------
A detector you cannot measure is a detector you cannot trust. This set lets us
compute real numbers — accuracy, precision, recall, and especially the
FALSE-POSITIVE rate (how often we wrongly flag a real person). Those numbers,
not a marketing "99%", are what an honest applicant-screening tool reports.

Honesty rules we followed when writing these samples
----------------------------------------------------
1. The human samples are varied on purpose: casual, formal, non-native English,
   terse bullet-points, buzzword-y new-grad, and a couple of *polished* writers
   who legitimately use em-dashes and rule-of-three phrasing. Real humans are
   not all "clean".
2. The AI samples are also varied: classic ChatGPT cover letters, Claude/Gemini
   cadence, old-style buzzword spam — AND two deliberately *lightly humanized*
   samples where the obvious tells were edited out. Those are SUPPOSED to be
   hard (they model an applicant who cleaned up their AI draft).
3. Because of (1) and (2) the measured accuracy is intentionally NOT 100%. The
   overlap is the whole point — it is where the fairness trade-off lives.

This is a STARTER set (~28 items). To publish a defensible number, expand it
with real, consented samples and keep the classes balanced. See eval/README.md.

Each record: {"id", "label": "human"|"ai", "style", "text"}
"""

from __future__ import annotations

from typing import Dict, List

SAMPLES: List[Dict[str, str]] = [
    # ======================================================================
    # HUMAN — genuine applicants (label = "human")
    # ======================================================================
    {
        "id": "h01",
        "label": "human",
        "style": "casual, concrete, uneven",
        "text": (
            "I fixed the checkout bug that lost us $4k a week. Took three days. "
            "At Zappos I rebuilt the returns queue in Go and cut latency from "
            "800ms to 90ms. Also mentored two interns. One shipped our first "
            "GraphQL endpoint. I like small teams. Less politics, more shipping. "
            "Hi Priya, saw the backend opening and figured I'd reach out. I write "
            "tests, I keep good runbooks, and I stick around when things break at 2am."
        ),
    },
    {
        "id": "h02",
        "label": "human",
        "style": "non-native English, honest, specific",
        "text": (
            "Dear Sir or Madam, I am writing to apply for the data analyst "
            "position. I have three years experience in my country working with "
            "SQL and Excel for a logistics company. I built the weekly report "
            "that management used for planning trucks and drivers. It saved maybe "
            "two hours every week for my team. I am hard working and I learn fast. "
            "I hope you give me a chance to show my skills. Thank you. Regards, Priya."
        ),
    },
    {
        "id": "h03",
        "label": "human",
        "style": "formal senior eng, uses ONE em-dash (hard case)",
        "text": (
            "I'm a backend engineer with five years on payment systems. At Stripe "
            "I owned the refund pipeline — it processed about 40,000 refunds a day "
            "and I cut its failure rate from 2% to under 0.3%. I write Python, Go, "
            "and a bit of Rust. I care about tests, clear logs, and on-call "
            "runbooks that actually help at 3am. Outside work I maintain a small "
            "open-source library for parsing bank statements."
        ),
    },
    {
        "id": "h04",
        "label": "human",
        "style": "new grad, mild career-services buzzwords",
        "text": (
            "Recent CS graduate passionate about building software that helps "
            "people. During my capstone I led a team of four to build a campus "
            "food-sharing app. We used React, Node, and MongoDB. I handled the "
            "backend and the deployment. The app had 300 signups in the first "
            "month. I'm looking for my first full-time role where I can keep "
            "learning from senior engineers."
        ),
    },
    {
        "id": "h05",
        "label": "human",
        "style": "career changer, healthcare -> tech, personal",
        "text": (
            "I spent six years as an ICU nurse before I taught myself to code "
            "during night shifts. My first project was a med-schedule tracker "
            "because our paper system kept failing. It's now used on my old ward. "
            "I know I'm not the typical candidate. But I've made life-or-death "
            "decisions under pressure, and debugging a flaky test does not scare "
            "me. I'd love to talk about the junior developer role."
        ),
    },
    {
        "id": "h06",
        "label": "human",
        "style": "trades, plain, specific",
        "text": (
            "Licensed electrician, 12 years. I wire new residential builds and do "
            "commercial panel upgrades. Ran crews of up to six on a hospital "
            "retrofit last year, finished two weeks early with zero safety "
            "incidents. I show up early, I clean up after myself, and I don't cut "
            "corners on grounding. Looking for a lead role closer to home. "
            "References from three general contractors available."
        ),
    },
    {
        "id": "h07",
        "label": "human",
        "style": "blunt short note",
        "text": (
            "Saw you need a data engineer. I built the whole ETL stack at my "
            "current job, from scratch, on a shoestring. Airflow, dbt, Snowflake. "
            "It moves about 2TB a day now and rarely breaks. I'm bored here and "
            "want a harder problem. That's it, that's the pitch. Resume attached."
        ),
    },
    {
        "id": "h08",
        "label": "human",
        "style": "academic, formal but specific",
        "text": (
            "I am a postdoctoral researcher in computational biology. My thesis "
            "characterized splice-variant errors in three cancer cell lines using "
            "a nanopore pipeline I wrote in Python and Nextflow. Two first-author "
            "papers came out of it. I am moving to industry because I want my work "
            "to reach patients faster than the grant cycle allows. I would bring "
            "rigor and a stubborn habit of validating everything twice."
        ),
    },
    {
        "id": "h09",
        "label": "human",
        "style": "sales, numbers-heavy, direct",
        "text": (
            "Closed $3.2M in new logos last year, 118% of quota. I own the full "
            "cycle from cold outreach to signature, mostly mid-market SaaS. My "
            "trick is boring: I do the research nobody else does before the first "
            "call. I lost a big deal in Q2 because I over-promised on timelines, "
            "and I've been careful about that since. Want to bring that hunger to "
            "an early-stage team."
        ),
    },
    {
        "id": "h10",
        "label": "human",
        "style": "designer, portfolio voice",
        "text": (
            "Product designer, mostly mobile. I redrew the onboarding for a "
            "fintech app and drop-off fell by a third; I can show you the before "
            "and after. I sketch on paper first, always. I'm picky about spacing "
            "and I will fight for a good empty state. Not a fan of dark patterns "
            "and I'll say so. Portfolio link is on my resume."
        ),
    },
    {
        "id": "h11",
        "label": "human",
        "style": "polished PM, em-dash + two triads (HARD, overlaps AI)",
        "text": (
            "I'm a product manager with eight years shipping consumer apps. At "
            "Spotify I owned onboarding — activation went from 31% to 44% over two "
            "quarters. I work closely with design, engineering, and research, and "
            "I care about shipping, measuring, and iterating. My last team "
            "launched three features that each moved retention. I'd love to bring "
            "that same discipline here."
        ),
    },
    {
        "id": "h12",
        "label": "human",
        "style": "quirky, humorous, specific",
        "text": (
            "Reformed physicist, now I make databases go fast. I once spent a "
            "weekend removing a single lock and tripled our write throughput; the "
            "team bought me tacos. I read query plans for fun, which my partner "
            "finds deeply weird. I'm allergic to meetings that should've been a "
            "Slack message. If you have a gnarly Postgres problem, I'm your person."
        ),
    },
    {
        "id": "h13",
        "label": "human",
        "style": "non-native, short, plain",
        "text": (
            "Hello, I want to apply for junior frontend job. I study web "
            "development two years by myself, using free courses. I make three "
            "projects with React, you can see on my GitHub. My English is not "
            "perfect but I understand documentation very well and I ask when I "
            "don't know. I am ready to work hard and learn from your team."
        ),
    },
    {
        "id": "h14",
        "label": "human",
        "style": "bootcamp grad, earnest, specific",
        "text": (
            "Career switcher, finished a 6-month bootcamp in May. Before that I "
            "managed a coffee shop for four years, so I know about rushes, "
            "inventory, and keeping calm when the espresso machine dies mid-shift. "
            "My capstone was a shift-swapping app for hourly workers, built with "
            "Django. Twelve of my old coworkers actually use it. I'm hungry for a "
            "junior role and I'll outwork anyone."
        ),
    },

    # ======================================================================
    # AI — assistant-written / heavily assisted (label = "ai")
    # ======================================================================
    {
        "id": "a01",
        "label": "ai",
        "style": "classic ChatGPT cover letter",
        "text": (
            "Dear Hiring Manager, I am writing to express my enthusiasm for the "
            "Software Engineer role at your esteemed organization. As a passionate "
            "and dedicated professional, I have always been drawn to the "
            "intersection of technology and real-world impact. Throughout my "
            "career, I have honed my ability to deliver scalable, robust, and "
            "innovative solutions. I am confident that my unique blend of skills "
            "would make me a valuable addition to your team. I look forward to the "
            "opportunity to contribute to your continued success."
        ),
    },
    {
        "id": "a02",
        "label": "ai",
        "style": "ChatGPT resume summary",
        "text": (
            "Results-driven and detail-oriented Software Engineer with a proven "
            "track record of delivering cutting-edge solutions. Passionate about "
            "leveraging innovative technologies to drive meaningful impact. Adept "
            "at designing, developing, and deploying scalable systems in "
            "fast-paced environments. Committed to excellence, continuous "
            "learning, and collaborating cross-functionally to exceed "
            "organizational goals and deliver best-in-class results."
        ),
    },
    {
        "id": "a03",
        "label": "ai",
        "style": "Claude-style polished cadence",
        "text": (
            "What draws me to this role is the opportunity to work at the "
            "intersection of craft and scale. Over the years, I have honed my "
            "ability to navigate the complexities of distributed systems — a "
            "testament to both my technical depth and my commitment to continuous "
            "growth. I thrive in environments where curiosity, collaboration, and "
            "rigor are valued, and I believe my background would make a meaningful "
            "contribution to your mission."
        ),
    },
    {
        "id": "a04",
        "label": "ai",
        "style": "Gemini-style, structured, triads + em-dashes",
        "text": (
            "As a Machine Learning Engineer, I bring a strong foundation in "
            "designing, building, and deploying production systems. My experience "
            "spans data pipelines, model training, and evaluation — always with an "
            "emphasis on reliability, scalability, and impact. I am passionate "
            "about turning ambiguous problems into clear, measurable outcomes, and "
            "I pride myself on writing clean, well-tested, and maintainable code. "
            "I would be thrilled to bring this blend of skills to your team."
        ),
    },
    {
        "id": "a05",
        "label": "ai",
        "style": "heavy em-dashes + triads, few buzzwords (representative)",
        "text": (
            "My background spans the full stack — frontend, backend, and "
            "infrastructure. I have built systems for authentication, billing, and "
            "analytics, and I enjoy the work of designing, implementing, and "
            "maintaining them end to end. I believe great engineering is about "
            "clarity, correctness, and care — not cleverness. I would welcome the "
            "chance to bring that philosophy to your team and to contribute to "
            "your organization's continued growth and success."
        ),
    },
    {
        "id": "a06",
        "label": "ai",
        "style": "LIGHTLY HUMANIZED AI (tells edited out — HARD, likely missed)",
        "text": (
            "I've spent about six years building web apps, mostly in Python and "
            "TypeScript. At my last job I rewrote our billing service and it cut "
            "support tickets by around 40 percent. I like working on problems that "
            "touch real users. I'm comfortable owning a feature from design to "
            "deploy, and I try to leave code better than I found it. I'd be glad to "
            "chat about how I could help your team."
        ),
    },
    {
        "id": "a07",
        "label": "ai",
        "style": "AI listing skills in rule-of-three",
        "text": (
            "I offer a comprehensive skill set spanning development, deployment, "
            "and documentation. My strengths include problem-solving, "
            "communication, and adaptability. I have delivered projects across web, "
            "mobile, and cloud platforms, using tools such as React, Node, and "
            "Kubernetes. I am eager, dependable, and driven, and I am confident I "
            "can design, build, and scale solutions that align with your goals, "
            "your vision, and your values."
        ),
    },
    {
        "id": "a08",
        "label": "ai",
        "style": "over-formal AI, transition-opener spam",
        "text": (
            "Furthermore, I possess extensive experience in software development. "
            "Moreover, I have consistently demonstrated an ability to deliver "
            "high-quality results. Additionally, my strong communication skills "
            "enable me to collaborate effectively. Consequently, I am well-suited "
            "for this position. Ultimately, I am confident that my expertise and "
            "dedication would prove to be a valuable asset to your esteemed "
            "organization in today's competitive landscape."
        ),
    },
    {
        "id": "a09",
        "label": "ai",
        "style": "AI for a non-tech (marketing) role",
        "text": (
            "As a passionate and results-driven marketing professional, I am "
            "excited to apply for this opportunity. I have a proven track record "
            "of crafting compelling narratives that resonate with target "
            "audiences and drive measurable engagement. Leveraging data-driven "
            "insights, I design campaigns that are both innovative and impactful. "
            "I thrive in dynamic, fast-paced environments and would be delighted "
            "to bring my creativity and strategic vision to your brand."
        ),
    },
    {
        "id": "a10",
        "label": "ai",
        "style": "old-style buzzword spam",
        "text": (
            "I am a dynamic, self-starting team player who leverages synergy to "
            "move the needle. I utilize cutting-edge, best-in-class methodologies "
            "to spearhead cross-functional initiatives and deliver robust, "
            "scalable solutions. My results-oriented, detail-oriented approach "
            "empowers me to think outside the box and drive value-add outcomes "
            "that align with strategic objectives and core competencies."
        ),
    },
    {
        "id": "a11",
        "label": "ai",
        "style": "AI new-grad cover letter",
        "text": (
            "Dear Hiring Committee, As a recent graduate with a strong foundation "
            "in computer science, I am thrilled to apply for your entry-level "
            "position. My academic journey has equipped me with a robust "
            "understanding of algorithms, data structures, and software design. I "
            "am a passionate, motivated, and quick learner, eager to contribute to "
            "your team while continuing to grow. I am confident that my dedication "
            "would make me a valuable addition to your organization."
        ),
    },
    {
        "id": "a12",
        "label": "ai",
        "style": "AI 'unique blend / valuable asset'",
        "text": (
            "I believe my unique blend of technical expertise and interpersonal "
            "skills sets me apart as a candidate. With a deep passion for "
            "innovation and a commitment to excellence, I consistently strive to "
            "exceed expectations. I am adept at navigating complex challenges and "
            "delivering solutions that create lasting value. I am eager to "
            "leverage my skills to become a valuable asset to your forward-thinking "
            "and mission-driven team."
        ),
    },
    {
        "id": "a13",
        "label": "ai",
        "style": "generic AI 'in today's world'",
        "text": (
            "In today's fast-paced and ever-evolving digital world, businesses "
            "require professionals who can adapt and thrive. I am precisely such a "
            "professional. With a comprehensive understanding of modern "
            "technologies and a passion for continuous improvement, I am "
            "well-positioned to make a meaningful impact. My holistic approach and "
            "keen attention to detail enable me to deliver seamless, end-to-end "
            "solutions that underscore my commitment to quality."
        ),
    },
    {
        "id": "a14",
        "label": "ai",
        "style": "LIGHTLY HUMANIZED AI #2 (HARD, likely missed)",
        "text": (
            "I'm a data analyst with four years of experience, mostly in retail. "
            "I built dashboards in Looker that the merchandising team checks every "
            "morning, and I automated a monthly report that used to take two days "
            "by hand. I'm comfortable with SQL and Python, and I'm learning dbt "
            "right now. I like making messy data usable for people who aren't "
            "technical. Happy to walk through my work anytime."
        ),
    },
]


def counts() -> Dict[str, int]:
    """Return how many human vs ai samples we have (for a quick balance check)."""
    out = {"human": 0, "ai": 0}
    for s in SAMPLES:
        out[s["label"]] = out.get(s["label"], 0) + 1
    return out
