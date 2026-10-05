(function () {
  "use strict";
  const dialog = document.getElementById("search-dialog");
  const input = document.getElementById("site-search-input");
  const status = document.getElementById("site-search-status");
  const results = document.getElementById("site-search-results");
  if (!dialog || !input || !status || !results || dialog.dataset.searchReady) return;
  dialog.dataset.searchReady = "true";
  let loading;
  let index;
  let documents = [];

  function loadScript(path) {
    return new Promise(function (resolve, reject) {
      let url;
      try {
        if (!path) throw new Error("Search resource path missing");
        url = new URL(path, document.baseURI);
        if (url.origin !== location.origin) throw new Error("Search resources must be local");
      } catch (error) { reject(error); return; }
      const script = document.createElement("script");
      script.src = url.href;
      script.onload = function () { resolve(); };
      script.onerror = function () { script.remove(); reject(new Error("Search resource unavailable")); };
      document.head.append(script);
    });
  }

  async function loadIndex() {
    if (index) return;
    if (loading) return loading;
    loading = (async function () {
      if (!window.elasticlunr) await loadScript(dialog.dataset.searchLibrary);
      if (!window.searchIndex) await loadScript(dialog.dataset.searchIndex);
      if (!window.elasticlunr || !window.searchIndex) throw new Error("Search resource invalid");
      const loaded = window.elasticlunr.Index.load(window.searchIndex);
      const store = window.searchIndex.documentStore && window.searchIndex.documentStore.docs;
      if (!store) throw new Error("Search documents unavailable");
      documents = Object.keys(store).map(function (ref) {
        const doc = store[ref];
        return {
          ref: ref,
          title: String(doc.title || "Untitled"),
          description: String(doc.description || ""),
          body: String(doc.body || ""),
          href: String(doc.id || doc.permalink || ref)
        };
      });
      index = loaded;
    })();
    try { await loading; } finally { loading = null; }
  }

  function snippet(doc, query) {
    const text = (doc.body || doc.description).replace(/\s+/g, " ");
    const position = text.toLowerCase().indexOf(query.toLowerCase());
    const start = position < 0 ? 0 : Math.max(0, position - 55);
    return (start ? "…" : "") + text.slice(start, start + 180) + (text.length > start + 180 ? "…" : "");
  }

  function render() {
    results.replaceChildren();
    const query = input.value.trim();
    if (!query) {
      status.textContent = "Search documentation, examples, and guides.";
      return;
    }
    if (!index) return;
    const scores = new Map();
    let hits;
    try {
      hits = index.search(query, { fields: { title: { boost: 5 }, description: { boost: 2 }, body: { boost: 1 } }, expand: true, bool: "AND" });
    } catch (_) {
      status.textContent = "Could not search this phrase. Try a shorter query.";
      return;
    }
    hits.forEach(function (hit) {
      scores.set(hit.ref, hit.score);
    });
    const literal = query.toLowerCase();
    const isIdentifier = /[\w][_.][\w]/.test(query);
    if (isIdentifier) {
      // Nested YAML paths may be rendered as separate words rather than a literal dotted field name.
      const words = query.replace(/[_.]+/g, " ").trim();
      index.search(words, { fields: { title: { boost: 5 }, description: { boost: 2 }, body: { boost: 1 } }, expand: true, bool: "AND" }).forEach(function (hit) {
        scores.set(hit.ref, Math.max(scores.get(hit.ref) || 0, hit.score * 0.8));
      });
    }
    // The prose stemmer splits identifiers; preserve exact dotted/underscored names with a small literal pass.
    documents.forEach(function (doc) {
      let bonus = 0;
      if (doc.title.toLowerCase().includes(literal)) bonus += 30;
      if (doc.description.toLowerCase().includes(literal)) bonus += 12;
      if (isIdentifier && doc.body.toLowerCase().includes(literal)) bonus += 8;
      if (bonus) scores.set(doc.ref, (scores.get(doc.ref) || 0) + bonus);
    });
    const matches = documents.filter(function (doc) { return scores.has(doc.ref); }).sort(function (a, b) {
      const priority = function (doc) { return /\/docs\//.test(doc.href) ? 1.2 : 1; };
      return scores.get(b.ref) * priority(b) - scores.get(a.ref) * priority(a);
    }).slice(0, 8);
    let displayed = 0;
    matches.forEach(function (doc) {
      let url;
      try { url = new URL(doc.href, document.baseURI); } catch (_) { return; }
      if (url.protocol !== "https:" && url.protocol !== "http:") return;
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = url.href;
      link.className = "search-result";
      const title = document.createElement("span");
      title.className = "search-result-title";
      title.textContent = doc.title;
      const description = document.createElement("span");
      description.className = "search-result-snippet";
      description.textContent = snippet(doc, query);
      link.append(title, description);
      item.append(link);
      results.append(item);
      displayed += 1;
    });
    status.textContent = displayed ? displayed + " result" + (displayed === 1 ? "" : "s") + ". Use arrow keys to browse." : "No results. Try a tool name or a shorter phrase.";
  }

  async function prepare() {
    if (!index) status.textContent = "Loading search…";
    try {
      await loadIndex();
      render();
    } catch (_) {
      status.textContent = "Search could not load. Reopen search to retry, or browse the documentation.";
    }
  }
  dialog.addEventListener("site:dialog-open", prepare);
  input.addEventListener("input", function () { if (index) render(); });
  dialog.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && dialog.open) {
      // Search inputs may consume Escape to clear their value before the native dialog can close.
      event.preventDefault();
      dialog.close();
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    const links = Array.from(results.querySelectorAll("a"));
    if (!links.length) return;
    const current = links.indexOf(document.activeElement);
    if (document.activeElement !== input && current < 0) return;
    event.preventDefault();
    if (current === 0 && event.key === "ArrowUp") input.focus();
    else {
      const next = current < 0 ? (event.key === "ArrowDown" ? 0 : links.length - 1) : Math.max(0, Math.min(links.length - 1, current + (event.key === "ArrowDown" ? 1 : -1)));
      links[next].focus();
    }
  });
  document.addEventListener("keydown", function (event) {
    if (event.defaultPrevented || event.repeat || event.isComposing || event.altKey || event.shiftKey || event.ctrlKey === event.metaKey || event.key.toLowerCase() !== "k") return;
    if (!window.AIAgentsSite) return;
    event.preventDefault();
    window.AIAgentsSite.openDialog(dialog.id, document.activeElement);
  });
})();
