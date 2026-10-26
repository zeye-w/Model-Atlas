# Deploy Model Atlas to Cloudflare Workers

Model Atlas uses Next.js 16. The Cloudflare target is powered by **vinext**
(Cloudflare's Vite-based compatibility layer). The standard Next.js setup is
preserved for local development.

This first release has no server-side database, secrets, or paid AI inference.
The live Linear Regression experiment runs locally in each visitor's browser.

## Before you begin

- Use Node.js 22 or newer and `npm ci` to install the locked dependencies.
- Sign in to a Cloudflare account with Workers enabled.
- Ensure that your GitHub branch includes `cloudflare.config.ts` and
  `vite.config.ts`. At the time this guide was created, the application is on
  `feat/phase1-foundation` (PR #1), **not** on `main`.
- Use the Worker name **`model-atlas`** in Cloudflare. If that name is already
  taken in your account, update the `name` in `cloudflare.config.ts` first;
  it must match the Cloudflare Worker name.

## Option A — connect GitHub in the Cloudflare dashboard

1. Go to **Workers & Pages → Create application → GitHub** and authorize the
   Cloudflare GitHub integration if requested.
2. Select `zeye-w/Model-Atlas`.
3. Select the branch containing the app. For a preview **before merging PR #1**,
   set **`feat/phase1-foundation`** as the production branch in the Worker
   build settings/branch control. Do not leave `main` selected while it lacks
   the app source. If your initial setup cannot select this branch, use
   **Option B** to deploy directly or merge the PR after reviewing it.
4. Set the Worker name to `model-atlas` and use these **custom build settings**:

   | Setting | Value |
   | --- | --- |
   | Root directory | `/` (repository root) |
   | Install | `npm ci` |
   | Build command | `npm run build:vinext` |
   | Deploy command | `npm run deploy:vinext:prebuilt` |
   | Production branch (before PR #1 merges) | `feat/phase1-foundation` |

   The deploy command uses vinext's Cloudflare Build Output instead of a
   conventional Wrangler entry point. **Do not use the default
   `npx wrangler deploy`** command for this setup.
5. Save and deploy. When the build succeeds, open the `*.workers.dev` URL
   shown under the Worker's deployments.

If you later merge PR #1 into `main`, switch the Worker's production branch
to `main` in **Settings → Builds → Branch control**.

### Preview builds for later PRs

When a production branch with the app is configured, enable **Preview Builds**
under **Settings → Builds → Branch control** for non-production branches.
The vinext CLI also supports explicit Worker Previews:

```bash
npm run preview:vinext -- pr-example
```

This command requires Cloudflare authentication and a Worker Preview-capable
Cloudflare project. **Do not confuse** `vinext-cloudflare deploy --preview`
with a branch Preview: that flag selects a build environment, not a Preview
deployment. Cloudflare's generic default `npx wrangler preview` has not been
validated for this vinext `cloudflare.config.ts` setup; use the supported
vinext preview command when configuring custom preview workflows.

## Option B — deploy once from your computer

This avoids the `main` branch issue when you only want a quick visual check.

```bash
git clone --branch feat/phase1-foundation https://github.com/zeye-w/Model-Atlas.git
cd Model-Atlas
npm ci
npx cf auth login
npm run deploy:vinext
```

After a successful deployment, Cloudflare returns a Worker deployment URL.
Cloudflare account authorization is required; neither a Cloudflare API token
nor a GitHub secret should be committed to this repository.

## Validate the deployment

- `/`: home page, hero artwork and navigation.
- `/atlas`: model taxonomy and relationships.
- `/models`: model library and category filter.
- `/models/cnn`: model detail and architecture explanation.
- `/lab/linear-regression`: change **seed**, **noise** and **learning rate**;
  run training and confirm the fitted line and loss curve update.
- Confirm that browser Back/Forward navigation and mobile layout work.

Useful local checks:

```bash
npm run typecheck
npm test
npm run build:vinext
npm run start:vinext
```

If Cloudflare reports `Missing entry-point to Worker script or to assets
directory`, the deploy command was probably left as `npx wrangler deploy`.
Use `npm run deploy:vinext:prebuilt` after `npm run build:vinext` instead.

The vinext/Cloudflare toolchain is evolving. Lockfile updates and another
compatibility check are recommended before production use.
