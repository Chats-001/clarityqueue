from clarityqueue.model import combine_text


def test_combine_text_strips_whitespace():
    assert combine_text(" title ", " body ") == "title body"


def test_route_prediction_is_known(triage_model):
    result = triage_model.predict("API token expired", "Requests return 401 authentication errors")
    assert result.route == "integration"


def test_priority_prediction_is_known(triage_model):
    result = triage_model.predict("Locked account", "All users are blocked and cannot sign in")
    assert result.priority == "critical"


def test_confidences_are_probabilities(triage_model):
    result = triage_model.predict("slow dashboard", "It needs five minutes to open")
    assert 0 <= result.route_confidence <= 1
    assert 0 <= result.priority_confidence <= 1


def test_route_signals_are_bounded(triage_model):
    result = triage_model.predict("duplicate charge", "two card entries for one renewal")
    assert len(result.signals) <= 5
    assert all(isinstance(signal, str) for signal in result.signals)


def test_predict_many_preserves_count(triage_model):
    routes, route_scores, priorities, priority_scores = triage_model.predict_many(
        ["lost authenticator device", "CSV columns shifted"]
    )
    assert len(routes) == len(route_scores) == len(priorities) == len(priority_scores) == 2
