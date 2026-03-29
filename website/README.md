# AI Agents Framework - Website

Source for [ai-agents.rs](https://ai-agents.rs), built with [Zola](https://www.getzola.org/).

## Prerequisites

Install Zola:

```sh
# via cargo
cargo install zola

# or download a prebuilt binary from
# https://github.com/getzola/zola/releases
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
