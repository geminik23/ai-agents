(function () {
  "use strict";
  if (window.AIAgentsSite) return;

  const dialogs = Array.from(document.querySelectorAll("dialog[data-site-dialog]"));
  const openers = new WeakMap();
  const suppressRestore = new WeakSet();

  function syncDialog(dialog) {
    document.querySelectorAll("[data-open-dialog]").forEach(function (button) {
      if (button.dataset.openDialog === dialog.id) {
        button.setAttribute("aria-expanded", String(dialog.open));
      }
    });
    document.documentElement.classList.toggle("dialog-open", dialogs.some(function (item) {
      return item.open;
    }));
  }

  function openDialog(id, opener) {
    const dialog = document.getElementById(id);
    if (!dialog || !dialogs.includes(dialog) || typeof dialog.showModal !== "function") return;
    if (dialog.open) return;
    dialogs.forEach(function (other) {
      if (other.open) {
        suppressRestore.add(other);
        other.close();
      }
    });
    openers.set(dialog, opener || document.activeElement);
    dialog.showModal();
    syncDialog(dialog);
    const initialFocus = dialog.querySelector("[autofocus], input, button, a[href]");
    if (initialFocus) initialFocus.focus();
    dialog.dispatchEvent(new CustomEvent("site:dialog-open"));
  }

  dialogs.forEach(function (dialog) {
    dialog.addEventListener("close", function () {
      syncDialog(dialog);
      if (suppressRestore.has(dialog)) {
        suppressRestore.delete(dialog);
        return;
      }
      const opener = openers.get(dialog);
      if (!dialogs.some(function (item) { return item.open; }) && opener && opener.isConnected && opener.getClientRects().length) {
        opener.focus();
      }
    });
    dialog.addEventListener("click", function (event) {
      // Native modal dialogs own Escape and focus containment; a backdrop click is outside their box.
      if (event.target === dialog) {
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
          dialog.close();
        }
      }
    });
    if (dialog.id !== "search-dialog") {
      const desktopAt = Number(dialog.dataset.desktopAt) || (dialog.id === "docs-menu-dialog" ? 1024 : 768);
      const desktop = window.matchMedia("(min-width: " + desktopAt + "px)");
      function closeOnDesktop() {
        if (desktop.matches && dialog.open) dialog.close();
      }
      desktop.addEventListener("change", closeOnDesktop);
      closeOnDesktop();
    }
    syncDialog(dialog);
  });

  document.addEventListener("click", function (event) {
    const opener = event.target.closest("[data-open-dialog]");
    if (opener) {
      event.preventDefault();
      openDialog(opener.dataset.openDialog, opener);
      return;
    }
    const closer = event.target.closest("[data-close-dialog]");
    const dialog = event.target.closest("dialog[data-site-dialog]");
    const link = event.target.closest("a[href]");
    const followsLink = link && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey;
    if (dialog && dialog.open && (closer || followsLink)) {
      // Hash navigation takes over focus/scroll; restoring the drawer opener afterwards would undo it.
      if (followsLink) suppressRestore.add(dialog);
      dialog.close();
    }
  });

  const themeToggle = document.getElementById("theme-toggle");
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const dark = document.getElementById("giallo-dark");
    const light = document.getElementById("giallo-light");
    if (dark) dark.disabled = theme === "light";
    if (light) light.disabled = theme !== "light";
    if (themeToggle) {
      themeToggle.setAttribute("aria-label", "Switch to " + (theme === "dark" ? "light" : "dark") + " theme");
    }
  }
  applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  const shortcutLabel = document.querySelector(".search-shortcut");
  if (shortcutLabel) shortcutLabel.textContent = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘ K" : "Ctrl K";
  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem("ai-agents-theme", next); } catch (_) { /* The switch works without persistence. */ }
    });
  }

  // Read code nodes instead of the decorated pre so line numbers and control labels never enter the clipboard.
  function codeText(code) {
    const clean = code.cloneNode(true);
    clean.querySelectorAll(".giallo-ln").forEach(function (number) { number.remove(); });
    clean.querySelectorAll(".code-prompt").forEach(function (prompt) {
      const following = prompt.nextSibling;
      if (following && following.nodeType === Node.TEXT_NODE && following.nodeValue.startsWith(" ")) {
        following.nodeValue = following.nodeValue.slice(1);
      }
      prompt.remove();
    });
    return clean.textContent;
  }

  document.querySelectorAll("pre.code-block, .docs-body pre, .page-content pre, .blog-content pre, .section-intro pre, article pre").forEach(function (pre) {
    const code = pre.querySelector("code");
    if (!code || pre.dataset.copyReady) return;
    pre.dataset.copyReady = "true";
    const wrapper = document.createElement("div");
    wrapper.className = "code-copy-wrap";
    pre.before(wrapper);
    wrapper.append(pre);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-copy-button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code");
    const status = document.createElement("span");
    status.className = "code-copy-status";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    wrapper.append(button, status);
    let resetTimer;
    button.addEventListener("click", async function () {
      clearTimeout(resetTimer);
      status.textContent = "";
      try {
        if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error("Clipboard unavailable");
        await navigator.clipboard.writeText(codeText(code));
        button.textContent = "Copied";
        status.textContent = "Code copied.";
      } catch (_) {
        button.textContent = "Copy";
        status.textContent = "Could not copy. Select the code and copy it manually.";
      }
      resetTimer = setTimeout(function () { button.textContent = "Copy"; }, 2500);
    });
  });

  document.querySelectorAll("[data-tabs]").forEach(function (group, groupIndex) {
    if (group.dataset.tabsReady) return;
    const list = group.querySelector("[data-tab-list], [role='tablist']");
    if (!list) return;
    const tabs = Array.from(list.querySelectorAll("[data-tab], [role='tab']"));
    const panels = tabs.map(function (tab) { return document.getElementById(tab.getAttribute("aria-controls")); });
    if (!tabs.length || panels.some(function (panel) { return !panel || !group.contains(panel); })) return;
    group.dataset.tabsReady = "true";
    list.setAttribute("role", "tablist");
    function select(index, focus) {
      tabs.forEach(function (tab, item) {
        tab.setAttribute("aria-selected", String(item === index));
        tab.tabIndex = item === index ? 0 : -1;
        tab.classList.toggle("is-active", item === index);
        panels[item].hidden = item !== index;
      });
      if (focus) tabs[index].focus();
    }
    tabs.forEach(function (tab, index) {
      if (!tab.id) tab.id = "site-tab-" + groupIndex + "-" + index;
      tab.setAttribute("role", "tab");
      panels[index].setAttribute("role", "tabpanel");
      panels[index].setAttribute("aria-labelledby", tab.id);
      panels[index].tabIndex = 0;
      tab.addEventListener("click", function () { select(index, false); });
      tab.addEventListener("keydown", function (event) {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
        else if (event.key === "Home") next = 0;
        else if (event.key === "End") next = tabs.length - 1;
        else return;
        event.preventDefault();
        select(next, true);
      });
    });
    const initial = tabs.findIndex(function (tab) { return tab.getAttribute("aria-selected") === "true"; });
    select(initial < 0 ? 0 : initial, false);
  });

  window.AIAgentsSite = Object.freeze({ openDialog: openDialog });
  if (dialogs.every(function (dialog) { return typeof dialog.showModal === "function"; })) {
    document.documentElement.classList.add("site-ready");
  }
})();
