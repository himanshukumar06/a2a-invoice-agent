# Vercel deployment

This version uses Vercel's native FastAPI detection. Do NOT add a catch-all rewrite.
Vercel recognizes `app.py` and the FastAPI `app` object automatically.

1. Push this folder to GitHub.
2. Import the repository into Vercel.
3. Set these environment variables:
   - `DATABASE_URL`
   - `AI_API_KEY`
   - `AI_MODEL` (for example `gpt-4.1-mini`)
   - `A2A_BEARER_TOKEN`
   - `BASE_URL` (optional; if omitted, the app derives `https://<production-domain>/a2a/`)
4. Deploy.
5. Test:
   `https://YOUR-DOMAIN.vercel.app/.well-known/agent-card.json`
6. Submit:
   `https://YOUR-DOMAIN.vercel.app/a2a/`

Do not commit `.env` or secrets.
