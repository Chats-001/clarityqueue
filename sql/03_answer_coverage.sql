SELECT date(created_at) AS event_date,
       count(*) AS answer_requests,
       avg(CAST(answered AS REAL)) AS answer_coverage,
       avg(confidence) AS mean_evidence_score
FROM inference_event
WHERE event_type = 'answer'
GROUP BY 1
ORDER BY 1;

