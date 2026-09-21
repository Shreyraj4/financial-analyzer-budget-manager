# Deploying (Railway + Supabase)

One Docker image serves the API and the dashboard (`/app`). The database is Supabase Postgres.

## Steps

1. **Supabase:** create a project, copy the connection string (Project Settings -> Database -> URI).
2. **Railway:** New Project -> Deploy from GitHub repo. It picks up `railway.json` and the root `Dockerfile`.
3. **Variables** (Railway -> Variables):
   - `DATABASE_URL` - the Supabase URI
   - `JWT_SECRET` - a long random string (never reuse the local one)
   - `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (a `:free` model works), or `ANTHROPIC_API_KEY`; with neither, reports use the built-in template
   - `CORS_ORIGINS` - only needed if the dashboard is hosted somewhere else; same-origin `/app` needs nothing
4. Deploy, then open `https://<your-app>.up.railway.app/app/`.

## What happens on deploy

- **Build:** installs requirements, then trains the five models (`categorization`, `forecasting`, `anomaly`,
  `clustering`, `recommendation`) from `data/synthetic/`. Trained `.joblib` files are not in git, so
  they must be built here. The build takes a few minutes.
- **Start:** `alembic upgrade head` (creates/updates tables), then `uvicorn` on `$PORT`.
- **Health check:** `GET /health`.

## Caveats

- The Dockerfile has not been built locally (Docker was not installed on the dev machine); the first
  Railway build is its first real test. If a build step fails, the log names the step.
- Free OpenRouter models are rate limited and may log prompts. Do not send real bank data through them.
- Categorization retraining from user labels writes to the container disk, which is wiped on redeploy.
