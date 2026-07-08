# Anvay Deployment Runbook

Target shape:

- Backend: Oracle VM, Docker Compose, Caddy TLS, FastAPI API, private Qdrant.
- Frontend: Vercel, same-origin `/api/anvay/*` proxy to the backend.
- Auth: Anvay password/session auth with secure HttpOnly cookies, CSRF checks,
  bootstrap admin, and Anvay-owned product membership authorization.
- LLMOps: Langfuse Cloud free tier when `LANGFUSE_*` keys are configured.

## 1. Prepare Oracle VM

Install Docker and the Compose plugin, then open only ports `80` and `443` in
Oracle Cloud firewall/security list. Do not expose Qdrant ports publicly.

Clone the backend repo on the VM:

```bash
git clone <backend-repo-url> anvay
cd anvay
```

Create production config:

```bash
cp anvay.prod.yaml.example anvay.yaml
cp .env.example .env
```

## 2. Required Backend Environment

Fill `.env`:

```bash
LLM_API_KEY=...
ANVAY_ENV=production
ANVAY_TOKEN_KEY=...
ANVAY_SECRET_KEY=...
ANVAY_ADMIN_API_KEY=...
ANVAY_BOOTSTRAP_ADMIN_EMAIL=you@example.com
ANVAY_BOOTSTRAP_ADMIN_PASSWORD=...
ANVAY_ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
ANVAY_API_DOMAIN=api.example.com
ANVAY_SKILLS_REPO=https://github.com/<org>/anvay-skills.git
ANVAY_SKILLS_REPO_TOKEN=...
ANVAY_ENABLE_LOCAL_FS_SOURCES=false
```

Generate `ANVAY_TOKEN_KEY`:

```bash
uv run python -c "from anvay.auth.token_cipher import TokenCipher; print(TokenCipher.generate_key())"
```

Generate `ANVAY_SECRET_KEY`, `ANVAY_ADMIN_API_KEY`, and
`ANVAY_BOOTSTRAP_ADMIN_PASSWORD`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Optional Langfuse:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
ANVAY_TRACE_CONTENT=false
```

Keep `ANVAY_TRACE_CONTENT=false` unless you explicitly want prompt/response
content in Langfuse.

### Client IP behind Caddy

With `ANVAY_ENV=production`, login rate limiting and access-request rate
limiting (`anvay/api/authz.py::client_ip`) trust the rightmost trusted
`X-Forwarded-For` hop closest to Caddy instead of the raw socket address.
Caddy's `reverse_proxy` (`deploy/Caddyfile`) appends the client address by
default, so no extra Caddy config is needed. If you put another proxy in front
of Caddy, ensure it forwards the original client chain correctly; otherwise all
clients behind it can collapse into one rate-limit bucket.

## 3. Start Backend

```bash
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml ps
```

Health check:

```bash
curl -i https://api.example.com/health
```

Expected:

```json
{ "status": "ok" }
```

Qdrant should not be reachable from public internet:

```bash
curl -i https://api.example.com:6333
```

That should fail.

## 4. Configure Vercel

In Vercel project settings, set:

```bash
ANVAY_API_URL=https://api.example.com
```

Deploy the frontend repo. Browser calls stay same-origin at `/api/anvay/*`;
the Vercel route handler forwards session cookies and CSRF headers to the
backend. Confirm the Vercel domain is listed in backend `ANVAY_ALLOWED_ORIGINS`.

## 5. First Login

Open the Vercel app and sign in with `ANVAY_BOOTSTRAP_ADMIN_EMAIL` plus
`ANVAY_BOOTSTRAP_ADMIN_PASSWORD`. On startup, the backend ensures this email is
an approved `admin` account and resets its password from the bootstrap env vars.
After initial setup, rotate or remove bootstrap credentials if you do not want
this behavior on future restarts. Other users can request access and remain
pending until an admin approves them.

## 6. Access Requests

Users can visit `/request-access`. Admin approves from:

```text
/admin/access
```

Approval assigns the user a Anvay app role.
Revoking a user blocks future backend access.

## 7. Operations

View logs:

```bash
docker compose -f docker-compose.prod.yml logs -f api
```

Restart:

```bash
docker compose -f docker-compose.prod.yml restart api
```

Upgrade:

```bash
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

Backup mounted Docker volumes regularly:

- `anvay_data`: SQLite registry/proposals/sessions/checkpoints.
- `anvay_skills`: local skills repo clone.
- `qdrant_data`: vector index.

## 8. Smoke Test Checklist

- `/health` returns `200`.
- Vercel app loads login page.
- Admin login succeeds.
- `/products` works only after login.
- Requests without a valid session or admin API bearer token fail.
- Non-admin users see only their own products.
- Product viewers cannot sync sources, run council, or approve proposals.
- Filesystem sources are rejected in production.
- Add source -> sync source -> SSE logs stream.
- Run council -> Langfuse trace appears when configured.
- Approve proposal -> skill Git commit/push succeeds before proposal status
  becomes `approved`.
