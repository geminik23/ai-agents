(function () {
  "use strict";
  const explorer = document.querySelector("[data-role-explorer]");
  if (!explorer) return;
  const form = explorer.querySelector("form");
  const levels = ["local", "role", "group", "router", "main"];
  const names = { local: "local override", role: "role leaf", group: "group default", router: "router default", main: "main alias" };
  // Mirrors RouterRolesConfig::select: a present alias wins; provider failures are not inheritance.
  function update() {
    const values = Object.fromEntries(levels.map(function (level) { return [level, form.elements.namedItem(level).value]; }));
    const selected = levels.find(function (level) { return values[level] !== ""; });
    explorer.querySelector("[data-role-result]").textContent = values[selected];
    explorer.querySelector("[data-role-reason]").textContent = "The " + names[selected] + " selects " + values[selected] + " for process.detect.";
    levels.forEach(function (level) {
      const row = explorer.querySelector('[data-role-level="' + level + '"]');
      row.classList.toggle("is-selected", selected === level);
      row.querySelector(".role-level-status").textContent = selected === level ? "Selected" : values[level] ? "Overridden" : "Inherits";
    });
    const yaml = [
      "name: RoleExplorer", 'system_prompt: "Answer clearly."', "llms:",
      "  main: { provider: openai, model: gpt-5.4-mini }",
      "  fast: { provider: openai, model: gpt-5.4-nano }",
      "  precise: { provider: openai, model: gpt-5.4-mini }", "llm:", "  default: " + values.main
    ];
    // Keep an empty mapping when all overrides inherit: omission would switch to legacy mode.
    if (!values.router && !values.group && !values.role) yaml.push("  router: {}");
    else {
      yaml.push("  router:");
      if (values.router) yaml.push("    default: " + values.router);
      if (values.group || values.role) {
        yaml.push("    process:");
        if (values.group) yaml.push("      default: " + values.group);
        if (values.role) yaml.push("      detect: " + values.role);
      }
    }
    yaml.push("process:", "  input:", "    - type: detect", "      config:");
    if (values.local) yaml.push("        llm: " + values.local);
    yaml.push("        detect: [language]", "tools: []");
    explorer.querySelector("[data-role-yaml]").textContent = yaml.join("\n");
  }
  form.addEventListener("change", update);
  form.addEventListener("submit", function (event) { event.preventDefault(); });
  form.addEventListener("reset", function () { setTimeout(update, 0); });
  explorer.querySelector("[data-role-inherit]").addEventListener("click", function () {
    levels.slice(0, -1).forEach(function (level) { form.elements.namedItem(level).value = ""; });
    update();
  });
  form.querySelectorAll("select, button").forEach(function (control) { control.disabled = false; });
  update();
})();
