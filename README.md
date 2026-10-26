# Model Atlas

**An interactive atlas for understanding and experimenting with machine learning models.**

Model Atlas is a public, research-oriented learning platform. It connects model families, architecture diagrams, mathematical explanations, references and small reproducible experiments in one interface.

## Phase 1 foundation

- Responsive site shell and light design system
- Searchable catalogue of eight representative models
- Algorithm family atlas with navigable relationships
- Shared model detail pages with overview, architecture, mathematics and references
- An actual, browser-side linear regression gradient descent laboratory
- Reproducible sample generation, loss history and JSON/SVG export

Other model laboratories are marked as planned; no animation is presented as real training.

## Quick start

Requires Node.js 20.9 or newer.

```bash
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000).

```bash
npm run typecheck
npm test
npm run build
```

## Deploying to Cloudflare Workers

The site is fully static: `next build` (with `output: "export"`) writes HTML and assets to `out/`, and `wrangler.jsonc` serves that directory as a static-assets-only Worker.

```bash
npm run preview   # build, then serve locally with `wrangler dev` (http://localhost:8787)
npm run deploy    # build, then `wrangler deploy` (run `npx wrangler login` first)
```

Continuous deployment is handled by `.github/workflows/deploy.yml`: pull requests run typecheck, tests and the build; pushes to `main` (or a manual run) also deploy. Add these repository secrets first:

| Secret | Value |
| --- | --- |
| `CLOUDFLARE_API_TOKEN` | API token with the **Edit Cloudflare Workers** template permissions |
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID |

Because the export is static, server-only features (request-time `searchParams`, route handlers, middleware, `next/image` optimization) are unavailable. Moving to the [OpenNext Cloudflare adapter](https://opennext.js.org/cloudflare) is the upgrade path if they are needed later.

## Project map

```text
src/app/                 Next.js routes and styles
src/components/          Shared UI and model visualizations
src/content/models.ts    Typed model registry
src/lib/                 Reproducible numerical experiments
tests/                   Algorithm tests
docs/                    Product specification and architecture
wrangler.jsonc           Cloudflare Workers configuration
```

## Scientific integrity

The linear regression lab runs gradient descent in the browser. It uses synthetic, seeded data and exports its settings and metrics. A future lab may instead use real inference or a precomputed trace, but every experience must identify its execution mode explicitly.

## License

The repository owner has not selected a license yet. Until one is added, the code is publicly visible but no open-source license is granted. See [Choose a License](https://choosealicense.com/) for guidance.
