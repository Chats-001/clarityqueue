"""Generate an original, seeded support-ticket benchmark from transparent templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clarityqueue.config import RANDOM_SEED

CASES = {
    "access": [
        (
            "locked out of {workspace}",
            "I cannot sign in after several attempts and the account says locked",
            "KB-ACCESS-01",
        ),
        (
            "lost MFA device",
            "My old phone is gone and I cannot enter the authenticator code",
            "KB-ACCESS-02",
        ),
        (
            "SSO keeps looping",
            "The identity provider succeeds but sends me back to login",
            "KB-ACCESS-03",
        ),
    ],
    "billing": [
        (
            "possible duplicate charge",
            "I see two {amount} card entries for the same renewal",
            "KB-BILL-01",
        ),
        (
            "wrong details on invoice",
            "The invoice has an incorrect company address and tax profile",
            "KB-BILL-02",
        ),
        (
            "renewal payment failed",
            "Our subscription is past due even though we updated the card",
            "KB-BILL-03",
        ),
    ],
    "data_sync": [
        (
            "warehouse data is late",
            "New {object} records have not synced for {hours} hours",
            "KB-SYNC-01",
        ),
        (
            "duplicate rows after reconnect",
            "The connector retry imported the same records more than once",
            "KB-SYNC-02",
        ),
        (
            "CSV fields missing",
            "Uploaded columns are blank or shifted after importing the file",
            "KB-SYNC-03",
        ),
    ],
    "integration": [
        (
            "webhook signature rejected",
            "Our endpoint reports an invalid HMAC signature",
            "KB-INT-01",
        ),
        (
            "API token expired",
            "Requests now return 401 and say the credential is expired",
            "KB-INT-02",
        ),
        (
            "too many requests",
            "The integration receives HTTP 429 during a traffic burst",
            "KB-INT-03",
        ),
    ],
    "performance": [
        (
            "dashboard became slow",
            "The analytics dashboard now needs {minutes} minutes to open",
            "KB-PERF-01",
        ),
        (
            "large export times out",
            "The {format} report fails before the download is ready",
            "KB-PERF-02",
        ),
        (
            "API latency spikes",
            "Response time intermittently jumps above {seconds} seconds",
            "KB-PERF-03",
        ),
    ],
}

PREFIXES = ["Please help:", "Customer report:", "Since this morning", "We noticed", "Question -"]
WORKSPACES = ["Acme North", "Demo Lab", "Retail Ops", "Finance Hub"]
OBJECTS = ["order", "customer", "invoice", "product"]
ROUTE_CONTEXT = {
    "access": "This appears related to user identity or account access.",
    "billing": "This appears related to subscription billing or payment.",
    "data_sync": "This appears related to moving data between systems.",
    "integration": "This appears related to an external API integration.",
    "performance": "This appears related to response speed or a timeout.",
}


def generate_tickets(n_per_case: int = 24, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    ticket_number = 1
    for route, cases in CASES.items():
        for title_template, body_template, article_id in cases:
            for _ in range(n_per_case):
                urgent = bool(rng.random() < 0.28)
                outage = bool(rng.random() < 0.08)
                if outage:
                    priority = "critical"
                elif urgent:
                    priority = "high"
                elif rng.random() < 0.35:
                    priority = "low"
                else:
                    priority = "medium"
                values = {
                    "workspace": rng.choice(WORKSPACES),
                    "amount": f"${rng.integers(20, 900)}",
                    "object": rng.choice(OBJECTS),
                    "hours": int(rng.integers(2, 30)),
                    "minutes": int(rng.integers(2, 12)),
                    "format": rng.choice(["PDF", "CSV"]),
                    "seconds": int(rng.integers(5, 45)),
                }
                impact = (
                    "All users are blocked."
                    if outage
                    else ("This affects a deadline today." if urgent else "One user is affected.")
                )
                context = ROUTE_CONTEXT[route] if rng.random() < 0.72 else ""
                records.append(
                    {
                        "ticket_id": f"CQ-{ticket_number:04d}",
                        "title": title_template.format(**values),
                        "description": (
                            f"{rng.choice(PREFIXES)} {body_template.format(**values)}. "
                            f"{context} {impact}"
                        ),
                        "route": route,
                        "priority": priority,
                        "article_id": article_id,
                    }
                )
                ticket_number += 1
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/generated/tickets.csv"))
    parser.add_argument("--per-case", type=int, default=24)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    frame = generate_tickets(args.per_case, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} original benchmark tickets to {args.output}")


if __name__ == "__main__":
    main()
