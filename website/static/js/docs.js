(function () {
  "use strict";
  const containers = Array.from(document.querySelectorAll("[data-toc]"));
  if (!containers.length || document.documentElement.dataset.tocReady) return;
  document.documentElement.dataset.tocReady = "true";
  const links = [];
  const headingLinks = new Map();
  containers.forEach(function (container) {
    container.querySelectorAll("a[href]").forEach(function (link) {
      let url;
      try { url = new URL(link.href, document.baseURI); } catch (_) { return; }
      if (url.pathname !== location.pathname || !url.hash) return;
      let id;
      try { id = decodeURIComponent(url.hash.slice(1)); } catch (_) { return; }
      const heading = document.getElementById(id);
      if (!heading) return;
      links.push(link);
      if (!headingLinks.has(heading)) headingLinks.set(heading, []);
      headingLinks.get(heading).push(link);
    });
  });
  const headings = Array.from(headingLinks.keys()).sort(function (a, b) {
    return a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1;
  });
  if (!headings.length) return;
  let active;
  function mark(heading) {
    if (active === heading) return;
    active = heading;
    links.forEach(function (link) {
      link.removeAttribute("aria-current");
      link.classList.remove("is-active");
      link.classList.remove("is-active-ancestor");
    });
    headingLinks.get(heading).forEach(function (link) {
      link.setAttribute("aria-current", "location");
      link.classList.add("is-active");
      const group = link.closest(".docs-toc-children");
      const parentLink = group && group.parentElement.querySelector(":scope > a");
      if (parentLink) parentLink.classList.add("is-active-ancestor");
    });
    // A reader's disclosure choices remain theirs; highlighting never closes or opens TOC details.
  }
  let scheduled = false;
  function update() {
    scheduled = false;
    const header = document.querySelector(".site-header");
    const threshold = (header ? header.getBoundingClientRect().bottom : 64) + 32;
    let current = headings[0];
    headings.forEach(function (heading) {
      if (heading.getBoundingClientRect().top <= threshold) current = heading;
    });
    mark(current);
  }
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(update);
  }
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule);
  window.addEventListener("hashchange", schedule);
  window.addEventListener("load", schedule);
  update();
})();
