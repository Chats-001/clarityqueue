SELECT route,
       count(*) AS tickets,
       avg(confidence) AS mean_confidence,
       rank() OVER (ORDER BY count(*) DESC) AS volume_rank
FROM inference_event
WHERE event_type = 'triage'
GROUP BY route
ORDER BY tickets DESC;

