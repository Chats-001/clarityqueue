"""Outcome-first portfolio dashboard for ClarityQueue."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from clarityqueue.api.schemas import TicketRequest
from clarityqueue.persistence import load_bundle

ARTIFACTS = Path("artifacts")
ROUTE_NAMES = {
    "access": "Access",
    "billing": "Billing",
    "data_sync": "Data sync",
    "integration": "Integration",
    "performance": "Performance",
}
OOD_QUERIES = [
    "What will the weather be tomorrow?",
    "Please approve my employee vacation request",
    "How should I calculate quarterly payroll tax?",
    "Book a hotel near the airport",
    "Our office coffee machine is leaking",
    "Write a marketing slogan for a new shoe",
    "What stocks should I buy this week?",
    "Translate this contract into French",
]
HUBS = pd.DataFrame(
    [
        ("New York", "Northeast", 40.7128, -74.0060, 18),
        ("Boston", "Northeast", 42.3601, -71.0589, 12),
        ("Atlanta", "Southeast", 33.7490, -84.3880, 17),
        ("Chicago", "Midwest", 41.8781, -87.6298, 17),
        ("Austin", "South", 30.2672, -97.7431, 16),
        ("Denver", "Mountain", 39.7392, -104.9903, 13),
        ("Seattle", "Northwest", 47.6062, -122.3321, 13),
        ("San Francisco", "West", 37.7749, -122.4194, 14),
    ],
    columns=["hub", "region", "lat", "lon", "review_capacity"],
)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate a 95% Wilson interval for a binomial rate."""
    rate = successes / total
    denominator = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denominator
    spread = z * sqrt((rate * (1 - rate) + z**2 / (4 * total)) / total) / denominator
    return centre - spread, centre + spread


def hub_scenario(rows: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, illustrative geography for capacity planning."""
    weighted_hubs = [0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7]
    scenario = rows.copy()
    ticket_numbers = scenario["ticket_id"].str.split("-").str[-1].astype(int)
    route_codes = scenario["route"].map({name: index for index, name in enumerate(ROUTE_NAMES)})
    slots = (ticket_numbers * 7 + route_codes * 3) % len(weighted_hubs)
    scenario["hub_index"] = slots.map(lambda value: weighted_hubs[int(value)])
    scenario = scenario.merge(HUBS.reset_index(names="hub_index"), on="hub_index")
    scenario["urgent"] = scenario["priority"].isin(["high", "critical"])
    scenario["needs_review"] = ~scenario["automation_candidate"]
    summary = (
        scenario.groupby(
            ["hub", "region", "lat", "lon", "review_capacity"], as_index=False
        )
        .agg(
            cases=("ticket_id", "size"),
            urgent_share=("urgent", "mean"),
            reviews=("needs_review", "sum"),
        )
    )
    summary["capacity_used"] = summary["reviews"] / summary["review_capacity"]
    return summary


def calibration_summary(rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Summarize whether model confidence matches observed accuracy."""
    frame = rows.copy()
    frame["band"] = pd.cut(
        frame["route_confidence"], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
    )
    calibration = (
        frame.groupby("band", observed=True)
        .agg(
            mean_confidence=("route_confidence", "mean"),
            observed_accuracy=("route_correct", "mean"),
            cases=("ticket_id", "size"),
        )
        .reset_index()
    )
    gap = abs(calibration["observed_accuracy"] - calibration["mean_confidence"])
    ece = float((gap * calibration["cases"]).sum() / len(frame))
    return calibration, ece


def text_map(rows: pd.DataFrame, fitted_model) -> tuple[pd.DataFrame, float]:
    """Project ticket text and discover issue clusters without using labels."""
    texts = (rows["title"] + " " + rows["description"]).tolist()
    vectorizer = fitted_model.route_model.named_steps["tfidf"]
    matrix = vectorizer.transform(texts)
    dimensions = min(20, matrix.shape[0] - 1, matrix.shape[1] - 1)
    dense = TruncatedSVD(n_components=dimensions, random_state=17).fit_transform(matrix)
    clusters = KMeans(n_clusters=5, n_init=20, random_state=17).fit_predict(dense)
    coordinates = TruncatedSVD(n_components=2, random_state=17).fit_transform(matrix)
    result = rows[["ticket_id", "title", "route", "route_correct"]].copy()
    result["SVD 1"] = coordinates[:, 0]
    result["SVD 2"] = coordinates[:, 1]
    result["cluster"] = [f"Cluster {value + 1}" for value in clusters]
    counts = pd.DataFrame({"cluster": clusters, "route": rows["route"].to_numpy()})
    dominant_counts = counts.groupby("cluster")["route"].value_counts().groupby(level=0).max()
    return result, float(dominant_counts.sum() / len(rows))


st.set_page_config(page_title="ClarityQueue", page_icon="CQ", layout="wide")
st.title("ClarityQueue")
st.subheader("From my first AI workshop project to an explainable decision system")
st.write(
    "I began this project during a **Break Through Tech workshop in December 2025**. "
    "I later rebuilt the idea to go beyond a single prediction: ClarityQueue now routes support "
    "cases, estimates urgency, retrieves a cited runbook, measures uncertainty, and knows when "
    "to ask a person for help."
)

if not (ARTIFACTS / "metrics.json").exists():
    st.error("No benchmark artifacts found. Run `make train` first.")
    st.stop()

model, index, metadata, metrics = load_bundle(ARTIFACTS)
predictions = pd.read_csv(ARTIFACTS / "test_predictions.csv")
quality = pd.read_csv(ARTIFACTS / "route_quality.csv")
route = metrics["route_model"]
retrieval = metrics["retrieval"]
operations = metrics["operational"]

first, second, third, fourth = st.columns(4)
first.metric("Correctly routed", f"{route['accuracy']:.1%}", "102 of 120 unseen cases")
second.metric("Balanced performance", f"{route['macro_f1']:.1%}", help="Macro F1")
third.metric("Safe to automate", f"{operations['automation_candidate_rate']:.1%}")
fourth.metric("Unrelated questions declined", f"{operations['out_of_domain_abstention_rate']:.1%}")

st.info(
    "**The outcome:** ClarityQueue could reduce manual sorting while keeping uncertain cases in "
    "human hands. Its clearest weakness is confusing symptom-heavy Access and Data sync tickets "
    "with Performance issues—so the next improvement is better training examples, "
    "not blind automation."
)

story, map_tab, ml_tab, rag_tab, explorer, journey = st.tabs(
    ["What it means", "Operations map", "ML + statistics", "RAG + safety", "Try it", "My journey"]
)

with story:
    left, right = st.columns(2)
    with left:
        st.markdown("#### From evaluation to responsible automation")
        total = len(predictions)
        routed = int(predictions["route_correct"].sum())
        candidates = int(predictions["automation_candidate"].sum())
        automated_correct = int(
            (
                predictions["automation_candidate"]
                & predictions["route_correct"]
                & predictions["correct"]
            ).sum()
        )
        funnel = go.Figure(
            go.Funnel(
                y=["Unseen cases", "Correct route", "Passed safety gate", "Correct end-to-end"],
                x=[total, routed, candidates, automated_correct],
                textinfo="value+percent initial",
                marker={"color": ["#4cc9f0", "#4895ef", "#4361ee", "#3a0ca3"]},
            )
        )
        funnel.update_layout(margin={"l": 15, "r": 15, "t": 15, "b": 15})
        st.plotly_chart(funnel, width="stretch")
        st.caption("This separates model accuracy from the smaller group safe enough to automate.")
    with right:
        st.markdown("#### Where the model needs help")
        confusion = pd.read_csv(ARTIFACTS / "route_confusion.csv", index_col=0)
        normalized = confusion.div(confusion.sum(axis=1), axis=0)
        labels = np.array(
            [
                [f"{value:.0%}<br>({confusion.iloc[row, col]})" for col, value in enumerate(line)]
                for row, line in enumerate(normalized.to_numpy())
            ]
        )
        heatmap = go.Figure(
            go.Heatmap(
                z=normalized.values,
                x=[ROUTE_NAMES[name] for name in normalized.columns],
                y=[ROUTE_NAMES[name] for name in normalized.index],
                colorscale="Blues",
                zmin=0,
                zmax=1,
                text=labels,
                texttemplate="%{text}",
                colorbar={"title": "Share"},
            )
        )
        heatmap.update_layout(xaxis_title="Predicted team", yaxis_title="Actual team")
        st.plotly_chart(heatmap, width="stretch")
        st.caption("Percentages make failure patterns comparable even when team volumes change.")

    st.markdown("#### My recommendation from the results")
    action_a, action_b, action_c = st.columns(3)
    action_a.success(
        "**Use it as an assistant**\n\nPre-sort confident cases and show cited guidance."
    )
    action_b.warning(
        "**Keep a human checkpoint**\n\nReview low-confidence and unfamiliar requests."
    )
    action_c.info(
        "**Improve the data next**\n\nCollect language that separates symptoms from root causes."
    )

with map_tab:
    st.markdown("#### Turning predictions into a staffing decision")
    st.caption(
        "Illustrative scenario: real held-out predictions are assigned to simulated US support "
        "hubs. "
        "Locations and staffing capacity are not customer data."
    )
    hub_summary = hub_scenario(predictions)
    map_figure = px.scatter_geo(
        hub_summary,
        lat="lat",
        lon="lon",
        size="cases",
        color="capacity_used",
        hover_name="hub",
        hover_data={
            "region": True,
            "cases": True,
            "reviews": True,
            "review_capacity": True,
            "urgent_share": ":.1%",
            "capacity_used": ":.1%",
            "lat": False,
            "lon": False,
        },
        color_continuous_scale="RdYlGn_r",
        range_color=[0, max(1.0, float(hub_summary["capacity_used"].max()))],
        scope="usa",
        labels={"capacity_used": "Review capacity used", "cases": "Cases"},
    )
    map_figure.update_geos(showlakes=True, lakecolor="#172033")
    map_figure.update_layout(height=520, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(map_figure, width="stretch")
    busiest = hub_summary.sort_values("capacity_used", ascending=False).iloc[0]
    map_a, map_b, map_c = st.columns(3)
    map_a.metric("Highest review pressure", busiest["hub"])
    map_b.metric("Review capacity used", f"{busiest['capacity_used']:.0%}")
    map_c.metric("Urgent case share", f"{busiest['urgent_share']:.0%}")
    st.info(
        "**What it means:** ML output becomes useful when it changes a real decision. Here, "
        "abstention volume becomes a staffing signal for where reviewers may be needed."
    )

with ml_tab:
    calibration, ece = calibration_summary(predictions)
    mapped_text, cluster_purity = text_map(predictions, model)
    rng = np.random.default_rng(17)
    correctness = predictions["route_correct"].astype(float).to_numpy()
    bootstrap = rng.choice(correctness, size=(2_000, len(correctness)), replace=True).mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("95% accuracy range", f"{lower:.1%}–{upper:.1%}", help="Bootstrap interval")
    metric_b.metric("Calibration error", f"{ece:.3f}", help="Lower is better")
    metric_c.metric("Cluster purity", f"{cluster_purity:.1%}", help="Unsupervised k-means")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Can I trust the confidence score?")
        calibration_plot = go.Figure()
        calibration_plot.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line={"dash": "dash"}
            )
        )
        calibration_plot.add_trace(
            go.Scatter(
                x=calibration["mean_confidence"],
                y=calibration["observed_accuracy"],
                mode="lines+markers+text",
                text=calibration["cases"].map(lambda value: f"n={value}"),
                textposition="top center",
                name="Observed",
                marker={"size": 11},
            )
        )
        calibration_plot.update_layout(
            xaxis={"title": "Mean predicted confidence", "range": [0, 1]},
            yaxis={"title": "Observed accuracy", "range": [0, 1]},
        )
        st.plotly_chart(calibration_plot, width="stretch")
        st.caption("Calibration connects a probability to an automation policy.")
    with right:
        st.markdown("#### Do issue themes appear without labels?")
        embedding = px.scatter(
            mapped_text,
            x="SVD 1",
            y="SVD 2",
            color="route",
            symbol="cluster",
            hover_data=["ticket_id", "title", "route_correct"],
            labels={"route": "Actual route"},
        )
        embedding.update_traces(marker={"size": 10, "opacity": 0.8})
        st.plotly_chart(embedding, width="stretch")
        st.caption("TF-IDF → SVD → k-means explores natural language groups before using labels.")

    st.markdown("#### How certain is each team-level result?")
    interval_rows = []
    for row in quality.itertuples(index=False):
        successes = round(row.recall * row.support)
        low, high = wilson_interval(successes, int(row.support))
        interval_rows.append(
            {
                "route": ROUTE_NAMES[row.route],
                "recall": row.recall,
                "lower": low,
                "upper": high,
                "support": int(row.support),
            }
        )
    intervals = pd.DataFrame(interval_rows).sort_values("recall")
    intervals["plus"] = intervals["upper"] - intervals["recall"]
    intervals["minus"] = intervals["recall"] - intervals["lower"]
    interval_plot = px.scatter(
        intervals,
        x="recall",
        y="route",
        error_x="plus",
        error_x_minus="minus",
        hover_data=["support"],
        range_x=[0, 1.05],
        labels={"recall": "Recall with 95% Wilson interval", "route": "Support team"},
    )
    interval_plot.update_traces(marker={"size": 13})
    st.plotly_chart(interval_plot, width="stretch")
    st.warning(
        "**What I learned:** a score is not a certainty. With only 24 held-out cases per team, "
        "even a perfect result has a wide range. Better evaluation data matters as much as "
        "modeling."
    )

with rag_tab:
    in_domain = predictions["top_score"].to_numpy()
    ood_scores = np.array([index.search(query, limit=1)[0].score for query in OOD_QUERIES])
    thresholds = np.linspace(0, max(float(in_domain.max()), float(ood_scores.max())) + 0.05, 35)
    threshold_rows = []
    for threshold in thresholds:
        threshold_rows.extend(
            [
                {
                    "threshold": threshold,
                    "answer_rate": float((in_domain >= threshold).mean()),
                    "population": "Support cases",
                },
                {
                    "threshold": threshold,
                    "answer_rate": float((ood_scores >= threshold).mean()),
                    "population": "Unrelated questions",
                },
            ]
        )
    threshold_frame = pd.DataFrame(threshold_rows)

    rag_a, rag_b, rag_c = st.columns(3)
    rag_a.metric("Correct runbook at #1", f"{retrieval['hit_at_1']:.1%}")
    rag_b.metric("Mean reciprocal rank", f"{retrieval['mean_reciprocal_rank']:.2f}")
    rag_c.metric("P95 retrieval time", f"{retrieval['p95_retrieval_latency_ms']:.2f} ms")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Relevant vs unrelated evidence scores")
        score_frame = pd.DataFrame(
            {
                "score": np.concatenate([in_domain, ood_scores]),
                "population": ["Support cases"] * len(in_domain)
                + ["Unrelated questions"] * len(ood_scores),
            }
        )
        st.plotly_chart(
            px.box(
                score_frame,
                x="population",
                y="score",
                color="population",
                points="all",
                labels={"score": "Top runbook cosine similarity", "population": ""},
            ),
            width="stretch",
        )
    with right:
        st.markdown("#### Coverage vs safety threshold")
        threshold_plot = px.line(
            threshold_frame,
            x="threshold",
            y="answer_rate",
            color="population",
            labels={"threshold": "Evidence threshold", "answer_rate": "Share answered"},
        )
        threshold_plot.add_vline(
            x=0.16,
            line_dash="dash",
            annotation_text="Current policy",
            annotation_position="top right",
        )
        threshold_plot.update_yaxes(range=[0, 1.05], tickformat=".0%")
        st.plotly_chart(threshold_plot, width="stretch")
    st.info(
        "**What it means:** a RAG system should not be judged only by whether it retrieves "
        "something. "
        "It should cite its source, remain fast, and decline when the evidence looks unrelated."
    )

with explorer:
    st.markdown("#### Follow one decision from text to cited guidance")
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
        st.write("**Words that influenced the route:**", ", ".join(triage.signals) or "None")
        if answer["answered"]:
            st.success(answer["answer"])
            st.caption(
                f"Source: {answer['citations'][0]['article_id']} — "
                f"{answer['citations'][0]['title']}"
            )
        else:
            st.warning(answer["answer"])

with journey:
    st.markdown(
        """
        #### My learning journey

        **December 2025 — the starting point**

        At a Break Through Tech workshop, I explored how data and machine learning could help
        classify a practical problem. It became my first data science and AI project.

        **The rebuild — asking better questions**

        I moved beyond “What is the accuracy?” and asked: Where does the model fail? How uncertain
        is the result? Can it explain a prediction? What happens when a question does not belong?

        **The current system — connecting disciplines**

        | Skill | What I implemented | Why it matters |
        |---|---|---|
        | Supervised ML | TF-IDF + logistic regression | Route and prioritize cases |
        | Unsupervised ML | SVD + k-means | Discover natural issue themes |
        | Statistics | Bootstrap + Wilson intervals | Communicate uncertainty |
        | Model governance | Calibration + abstention | Decide when a human should step in |
        | RAG | Retrieval + citations | Ground guidance in runbooks |
        | Geospatial analytics | Simulated capacity map | Connect predictions to staffing |
        | Software engineering | FastAPI, tests, Docker, CI | Make it usable and reproducible |

        **What I would do next**

        Gather real, consented support data; audit fairness across customer groups; measure drift;
        calibrate probabilities on a larger validation set; and run a shadow deployment before
        allowing any automated routing.

        This synthetic benchmark demonstrates my learning and engineering process. It is not a
        claim about production customer performance.
        """
    )
    st.json({"model_version": metadata["model_version"], **metrics["dataset"]})
