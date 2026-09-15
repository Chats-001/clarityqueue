WITH daily AS (
    SELECT date(created_at) AS event_date,
           event_type,
           count(*) AS requests,
           avg(latency_ms) AS mean_latency_ms
    FROM inference_event
    GROUP BY 1, 2
)
SELECT *, sum(requests) OVER (PARTITION BY event_date) AS total_daily_requests
FROM daily
ORDER BY event_date DESC, event_type;

