import pandas as pd

from clarityqueue.config import PRIORITIES, ROUTES
from scripts.generate_data import CASES, generate_tickets


def test_same_seed_is_reproducible():
    pd.testing.assert_frame_equal(generate_tickets(2, 10), generate_tickets(2, 10))


def test_different_seed_changes_examples():
    assert not generate_tickets(2, 10).equals(generate_tickets(2, 11))


def test_requested_case_count():
    assert len(generate_tickets(3)) == sum(len(cases) for cases in CASES.values()) * 3


def test_all_routes_are_represented(tickets):
    assert set(tickets["route"]) == set(ROUTES)


def test_all_priorities_are_represented(tickets):
    assert set(tickets["priority"]) == set(PRIORITIES)


def test_ticket_ids_are_unique(tickets):
    assert tickets["ticket_id"].is_unique


def test_ticket_text_is_populated(tickets):
    assert tickets["title"].str.len().min() > 3
    assert tickets["description"].str.len().min() > 10
