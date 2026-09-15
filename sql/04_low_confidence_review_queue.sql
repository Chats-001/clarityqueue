SELECT event_id, created_at, event_type, route, priority, confidence, metadata_json
FROM inference_event
WHERE confidence < 0.65 OR answered = 0
ORDER BY confidence ASC, created_at ASC;

