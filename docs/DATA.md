# Data design

The repository includes 15 original fictional runbooks in `data/source/knowledge_base.csv`. A seeded
generator creates 360 fictional support tickets from documented issue families, variable entities,
impact phrases, and optional route context. No customer or third-party data is used.

Generated ticket fields are `ticket_id`, `title`, `description`, `route`, `priority`, and
`article_id`. Generated CSVs are ignored because `python -m scripts.generate_data` recreates them.

This dataset is designed to exercise an end-to-end workflow, not to mimic the full linguistic,
temporal, cultural, or privacy complexity of production support data.

