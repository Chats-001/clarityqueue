"""Results-first Streamlit experience for ClarityQueue."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clarityqueue.api.schemas import TicketRequest
from clarityqueue.persistence import load_bundle

ARTIFACTS = Path("artifacts")

st.set_page_config(page_title="ClarityQueue", page_icon="CQ", layout="wide")
st.title("ClarityQueue")
st.subheader("Turn support requests into clear routes and evidence-backed next steps")
st.write(
    "ClarityQueue predicts where a case belongs, estimates urgency, retrieves a matching runbook, "
    "and declines to answer when the evidence is weak. Every headline number below comes from a "
    "held-out benchmark—not from the training examples."
)

if not (ARTIFACTS / "metrics.json").exists():
    st.error("No benchmark artifacts found. Run `python -m scripts.train` first.")
    st.stop()

model, index, metadata, metrics = load_bundle(ARTIFACTS)
predictions = pd.read_csv(ARTIFACTS / "test_predictions.csv")

route = metrics["route_model"]
retrieval = metrics["retrieval"]
operations = metrics["operational"]

first, second, third, fourth = st.columns(4)
first.metric(
    "Correct routing", f"{route['accuracy']:.1%}", help="Exact route accuracy on held-out tickets"
)
second.metric("Evidence found in top 3", f"{retrieval['hit_at_3']:.1%}")
third.metric("Safe answer coverage", f"{retrieval['answer_coverage']:.1%}")
fourth.metric(
    "Automated path accuracy",
    f"{operations['end_to_end_accuracy_when_automated']:.1%}",
    help="Both route and top evidence are correct among high-confidence automation candidates",
)

st.info(
    f"In plain English: the model tested {metrics['dataset']['held_out_tickets']} unseen cases. "
    f"It routed {route['accuracy']:.1%} correctly and placed the expected runbook in the top three "
    f"for {retrieval['hit_at_3']:.1%}. Low-evidence questions are intentionally sent to a person."
)

overview, tradeoffs, explorer, methodology = st.tabs(
    ["Outcome overview", "Confidence & tradeoffs", "Try a case", "How it works"]
)

with overview:
    left, right = st.columns(2)
    quality = pd.read_csv(ARTIFACTS / "route_quality.csv")
    with left:
        st.markdown("#### Route quality by team")
        figure = px.bar(
            quality,
            x="route",
            y=["precision", "recall", "f1"],
            barmode="group",
            range_y=[0, 1],
            labels={"value": "Score", "route": "Support team", "variable": "Metric"},
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "A route is useful only when it is both precise and able to find most relevant cases."
        )
    with right:
        st.markdown("#### What the router confused")
        confusion = pd.read_csv(ARTIFACTS / "route_confusion.csv", index_col=0)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=confusion.values,
                x=confusion.columns,
                y=confusion.index,
                colorscale="Blues",
                text=confusion.values,
                texttemplate="%{text}",
            )
        )
        heatmap.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(heatmap, width="stretch")
        st.caption(
            "The diagonal shows correct routes; off-diagonal cells reveal specific failure modes."
        )

with tradeoffs:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confidence separates easier and harder cases")
        confidence = px.histogram(
            predictions,
            x="route_confidence",
            color="route_correct",
            nbins=12,
            barmode="overlay",
            labels={"route_confidence": "Route confidence", "route_correct": "Correct route"},
        )
        st.plotly_chart(confidence, width="stretch")
    with right:
        st.markdown("#### Answer coverage vs evidence threshold")
        rows = []
        for threshold in [value / 100 for value in range(5, 81, 5)]:
            selected = predictions[predictions["top_score"] >= threshold]
            rows.append(
                {
                    "threshold": threshold,
                    "coverage": len(selected) / len(predictions),
                    "accuracy_when_answered": selected["correct"].mean() if len(selected) else None,
                }
            )
        curve = pd.DataFrame(rows)
        st.plotly_chart(
            px.line(
                curve,
                x="coverage",
                y="accuracy_when_answered",
                markers=True,
                labels={
                    "coverage": "Questions answered",
                    "accuracy_when_answered": "Top evidence correct",
                },
            ),
            width="stretch",
        )
    st.warning(
        "Coverage is not the same as quality. Raising the evidence threshold answers fewer "
        "questions "
        "but gives a human reviewer more of the uncertain cases."
    )

with explorer:
    title = st.text_input("Case title", value="Webhook signature rejected")
    description = st.text_area(
        "What happened?", value="Our production endpoint reports an invalid HMAC signature."
    )
    if st.button("Analyze case", type="primary"):
        request = TicketRequest(title=title, description=description)
        triage = model.predict(request.title, request.description)
        answer = index.answer(f"{request.title} {request.description}", route=triage.route)
        a, b, c = st.columns(3)
        a.metric("Suggested team", triage.route.replace("_", " ").title())
        b.metric("Urgency", triage.priority.title())
        c.metric("Evidence score", f"{answer['confidence']:.2f}")
        st.write("Routing signals:", ", ".join(triage.signals) or "No strong token signals")
        if answer["answered"]:
            st.success(answer["answer"])
            st.caption(
                f"Source: {answer['citations'][0]['article_id']} — "
                f"{answer['citations'][0]['title']}"
            )
        else:
            st.warning(answer["answer"])

with methodology:
    st.markdown(
        """
        #### A deliberately small, inspectable system

        1. A seeded generator creates support cases from documented templates.
        2. TF-IDF logistic regression predicts the support route and priority.
        3. A separate TF-IDF index ranks 15 operational runbooks.
        4. The response composer quotes steps only from the top retrieved runbook.
        5. Below the evidence threshold, the system abstains and asks for human review.

        This is retrieval-augmented response generation without a hosted LLM. That keeps the demo
        reproducible and makes retrieval quality, citations, latency, and abstention directly
        measurable. The synthetic benchmark demonstrates engineering methodology; it is not a claim
        about performance on real customer data.
        """
    )
    st.json({"model_version": metadata["model_version"], **metrics["dataset"]})
