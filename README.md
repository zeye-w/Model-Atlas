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

## Cloudflare Workers preview

The site can also build and run on Cloudflare Workers through **vinext**. This is
an additional deployment target; the original Next.js commands still work.

```bash
npm ci
npm run build:vinext
npm run start:vinext
```

For the first hosted preview, see [Cloudflare deployment guide](docs/DEPLOY_CLOUDFLARE.md).
Cloudflare Git builds must use the vinext build and deploy commands, not the
default `next build` and `wrangler deploy` pair.

## Project map

```text
src/app/                 Next.js routes and styles
src/components/          Shared UI and model visualizations
src/content/models.ts    Typed model registry
src/lib/                  Reproducible numerical experiments
tests/                   Algorithm tests
docs/                    Product specification and architecture
```

## Scientific integrity

The linear regression lab runs gradient descent in the browser. It uses synthetic, seeded data and exports its settings and metrics. A future lab may instead use real inference or a precomputed trace, but every experience must identify its execution mode explicitly.

## License

The repository owner has not selected a license yet. Until one is added, the code is publicly visible but no open-source license is granted. See [Choose a License](https://choosealicense.com/) for guidance.
