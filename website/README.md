# AI Agents Framework - Website

Source for [ai-agents.rs](https://ai-agents.rs), built with [Zola](https://www.getzola.org/).

## Prerequisites

Python **3.11 or newer** generates curated example pages using the standard library.
The website build does not execute agents or contact model providers.

Install Zola **0.23.6**, matching the CI pin. Earlier versions do not support the site's `skip_content_templating` configuration.

```sh
# via cargo
cargo install zola --version 0.23.6 --locked

# or download a prebuilt binary from
# https://github.com/getzola/zola/releases/tag/v0.23.6
```

## Build & Serve

The `build.sh` script syncs `examples/README.md` into the site content before running Zola. Always use it instead of calling `zola` directly.

```sh
# Build for production (output goes to public/)
./website/build.sh build

# Start the local dev server (live-reload on http://127.0.0.1:1111)
./website/build.sh serve

# Pass extra flags to zola
./website/build.sh serve -i 0.0.0.0 -p 8080
```

> **Note:** Run from the project root or from inside `website/`. The script > resolves paths relative to itself.

## Deployment

Push to `main` and Cloudflare Pages (or your CI) runs `./website/build.sh build`.
The output directory is `website/public/`.

### Search metadata and canonical URLs

`templates/partials/seo.html` renders the HTML title, description, canonical URL,
Open Graph metadata, and Twitter summary card metadata from one source. Content
uses its own title and description; `[extra].seo_title` can specialize a search
title without changing the visible heading. The homepage uses `extra.home_title`
and the site description from `config.toml`, and includes `WebSite` JSON-LD.
Sharing metadata currently describes the page in text without a social image.

Production builds use the HTTPS `base_url`. Keep it set to the public deployment
URL. Local serve builds use their local URL. Archive pages after page one have
their own canonical URLs and numbered titles. `templates/sitemap.xml` excludes
only the `/page/1/` redirect aliases created by the default pagination path;
later archive pages remain included. Revisit that filter if `paginate_path`
changes. Last-modified dates come from content dates, not the build time.

HTTP-to-HTTPS redirects are a hosting setting, not an HTML template feature.
For a Cloudflare-proxied domain, enable **SSL/TLS → Edge Certificates → Always
Use HTTPS**, or configure an equivalent permanent redirect at the actual host.
See [Cloudflare's HTTPS redirect documentation](https://developers.cloudflare.com/ssl/edge-certificates/additional-options/always-use-https/).
Preserve the requested path and query string. Verify after deploying:

```sh
curl -I 'http://ai-agents.rs/docs/yaml-reference/?source=redirect-check'
curl -I 'https://ai-agents.rs/docs/yaml-reference/'
```

The HTTP response should permanently redirect to the matching HTTPS URL, and the
HTTPS content URL should return 200. This repository does not configure the
hosting account or enable that setting automatically.

Validate generated metadata and links before deployment. After deployment,
inspect the homepage and key reference URLs in Search Console; use the live test
to check accessibility and optionally request indexing for updated pages.
Record the deployment date and compare equivalent reporting periods after Google
has recrawled. Track document indexing, relevant query impressions and clicks;
metadata changes do not guarantee indexing or a particular ranking.

## Project Layout

```
website/
├── build.sh           # Sync + build entry point
├── config.toml        # Zola configuration
├── content/           # All pages & posts as Markdown
├── sass/style.scss    # Site-wide styling
├── static/            # Static assets (images, fonts, etc.)
├── templates/         # Tera HTML templates
└── public/            # Generated site (git-ignored)
```

## Content Guide

| Content | Where to edit | Rebuilt automatically? |
|---------|---------------|----------------------|
| Examples | `examples/README.md` (project root) | Yes, via `build.sh` |
| Docs | `website/content/docs/*.md` | No - edit directly |
| Blog | `website/content/blog/*.md` | No - edit directly |
| Styles | `website/sass/style.scss` | Yes, Zola compiles Sass |
| Templates | `website/templates/*.html` | Yes, on `zola serve` |

## Adding a Blog Post

Create a new `.md` file in `content/blog/`:

```sh
cat > website/content/blog/my-post.md << 'EOF'
+++
title = "My New Post"
date = 2025-07-01
description = "A short summary."
template = "blog-page.html"
[taxonomies]
tags = ["update"]
+++

Post content goes here.
EOF
```

## Styling

Global styles live in `sass/style.scss`. Zola compiles Sass automatically.
The site uses CSS custom properties for dark/light theming - edit the `:root`
and `[data-theme="light"]` blocks to change colors.

Zola generates `giallo-light.css` and `giallo-dark.css` from the configured
high-contrast syntax themes during each build. Do not add copies to `static/`:
static assets are copied after generation and would overwrite those styles.

## Framework experience content

`catalog/recipes.toml` supplies the curated example metadata. The build wrapper
generates `content/examples/recipe-*.md` from the original YAML files under
`examples/yaml/`. Edit the catalog or original examples, then rerun the wrapper.
These generated pages and the README-derived examples index are ignored by Git.
When serving locally, restart the wrapper after changing recipe metadata or source
YAML; Zola watches website sources but does not rerun the generator itself.

The documentation hub uses `docs-index.html`. Learning guides set
`extra.learning_guide = true`; existing reference pages keep their URLs and anchors.
The role selection explorer at `/explore/roles/` explains mapping-form inheritance
without making model requests. Its generated YAML uses declared aliases and the
same local → role → group → router → main order as the Rust resolver.

The approval recipe displays an assertion-backed walkthrough of the public HITL mocked
suite. It is a recorded evaluation, with model, HTTP, and approval fixtures; it is
not a live session, raw event trace, or provider benchmark. To refresh it from the
current checkout:

```sh
cargo run -p ai-agents-cli --locked -- eval \
  --scenarios examples/eval/mocked/hitl/hitl_basic_mocked.yaml \
  --output target/website-demo/hitl
python3 website/scripts/capture_execution_demo.py target/website-demo/hitl/summary.json
sh website/build.sh build
```

`catalog/execution-demo.json` contains only an allowlisted summary: source revision,
public input hashes, report hash, passing assertion names, and verified execution
flags. Full reports, raw responses, prompts, paths to local reports, and approval
arguments are not copied into the site. The build verifies the public fixture and
agent hashes and fails if they have changed since capture. The recorded revision
identifies the evaluated runtime; regenerate after relevant runtime changes.

Shared presentation lives in `_shell.scss`; homepage, examples, learning, and
interactive explanations have separate SCSS partials. All primary content remains
readable without JavaScript. JavaScript enhances example filters, recorded outcome
tabs, and the role explorer.

The homepage pairs complete, downloadable definitions in `static/agents/` with
Assistant, Translator, and Tutor conversations. `catalog/home-agents.toml` contains
the public fixed-response fixtures. The recorded replies in
`catalog/home-conversations.json` come from running those definitions through the
normal runtime with a mock model. The visible demo is not a live provider session;
its explanation is available under "About these examples". Refresh after changing
any definition or fixture:

```sh
python3 website/scripts/home_agent_demos.py --record
sh website/build.sh build
```

Recording requires the Rust toolchain and locally cached locked dependencies. The
normal build only checks hashes and does not run a model or agent. Recording
exports only the declared public exchange, not full evaluation reports or local
paths. The expansion panels below the homepage demo are independent YAML excerpts;
the multi-agent panel also needs the linked specialist definition.

## Homepage background

The hero's static dot grid and soft background color live in
`templates/partials/hero-background.html` and `sass/_hero-background.scss`.
The decoration uses CSS only, with theme and mobile adjustments. It has no
pointer effects, animations, scripts, or interactive controls. Forced colors
hides the decoration; foreground content and controls remain independent.
