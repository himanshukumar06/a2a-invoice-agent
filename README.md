# A2A Invoice Agent on Vercel

Stateless FastAPI function (`api/index.py`) + Postgres. See the chat reply for the step-by-step deploy.

Local test (needs a Postgres):  `TEST_DATABASE_URL=postgresql://u:p@localhost/db python tests_smoke.py`
Tables are created automatically on first request (schema is in `SCHEMA` inside api/index.py).
