# Nexus Deployment Runbook

Target shape:

- Backend: Oracle VM, Docker Compose, Caddy TLS, FastAPI API, private Qdrant.
- Frontend: Vercel, `NEXT_PUBLIC_NEXUS_API=https://api.example.com`.
- Auth: built-in email/password, Argon2id password hashes, secure HttpOnly
  session cookies, CSRF header on unsafe requests.
- LLMOps: Langfuse Cloud free tier when `LANGFUSE_*` keys are configured.

## 1. Prepare Oracle VM

Install Docker and the Compose plugin, then open only ports `80` and `443` in
Oracle Cloud firewall/security list. Do not expose Qdrant ports publicly.

Clone the backend repo on the VM:

```bash
git clone <backend-repo-url> nexus
cd nexus
```

Create production config:

```bash
cp nexus.prod.yaml.example nexus.yaml
cp .env.example .env
```

## 2. Required Backend Environment

Fill `.env`:

```bash
DEEPINFRA_API_KEY=...
GITHUB_TOKEN=...
NEXUS_TOKEN_KEY=...
NEXUS_SECRET_KEY=...
NEXUS_ADMIN_API_KEY=...
NEXUS_BOOTSTRAP_ADMIN_EMAIL=you@example.com
NEXUS_BOOTSTRAP_ADMIN_PASSWORD=<long-password>
NEXUS_ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
NEXUS_API_DOMAIN=api.example.com
NEXUS_SKILLS_REPO=https://github.com/<org>/nexus-skills.git
```

Generate `NEXUS_TOKEN_KEY`:

```bash
uv run python -c "from nexus.auth.token_cipher import TokenCipher; print(TokenCipher.generate_key())"
```

Generate `NEXUS_SECRET_KEY` and `NEXUS_ADMIN_API_KEY`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Optional Langfuse:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
NEXUS_TRACE_CONTENT=false
```

Keep `NEXUS_TRACE_CONTENT=false` unless you explicitly want prompt/response
content in Langfuse.

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
{"status":"ok"}
```

Qdrant should not be reachable from public internet:

```bash
curl -i https://api.example.com:6333
```

That should fail.

## 4. Configure Vercel

In Vercel project settings, set:

```bash
NEXT_PUBLIC_NEXUS_API=https://api.example.com
```

Deploy the frontend repo. Confirm the Vercel domain is listed in backend
`NEXUS_ALLOWED_ORIGINS`.

## 5. First Login

Open the Vercel app and sign in with:

```text
NEXUS_BOOTSTRAP_ADMIN_EMAIL
NEXUS_BOOTSTRAP_ADMIN_PASSWORD
```

The backend creates this admin user on first boot. Passwords are stored as
Argon2id encoded hashes with per-password random salts.

## 6. Access Requests

Users can visit `/request-access`. Admin approves from:

```text
/admin/access
```

Approval requires assigning a temporary password. Revoking a user also revokes
active sessions.

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

- `nexus_data`: SQLite registry/proposals/sessions/checkpoints.
- `nexus_skills`: local skills repo clone.
- `qdrant_data`: vector index.

## 8. Smoke Test Checklist

- `/health` returns `200`.
- Vercel app loads login page.
- Admin login succeeds.
- `/products` works only after login.
- Unsafe requests fail without `X-Nexus-CSRF`.
- Add source -> sync source -> SSE logs stream.
- Run council -> Langfuse trace appears when configured.
- Approve proposal -> skill Git commit/push succeeds before proposal status
  becomes `approved`.
