# PDF corpus fixtures

Place representative PDFs here when the validation corpus is available.

Expected categories:

- `simple_text.pdf`
- `math_course.pdf`
- `scientific_article_two_columns.pdf`
- `slides_export.pdf`
- `tables_dense.pdf`
- `scanned_or_low_quality.pdf`
- `figures_and_diagrams.pdf`
- `french_course.pdf`
- `english_paper.pdf`

Each PDF can be paired with a `<stem>.expected.json` file containing minimal
quality thresholds such as `min_score`, `min_blocks`, `expected_types`, and
`allow_warnings`.
