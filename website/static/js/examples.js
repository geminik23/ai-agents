(() => {
    "use strict";
    const gallery = document.querySelector("[data-example-gallery]");
    if (!gallery) return;
    const form = gallery.querySelector("[data-example-filters]");
    const count = gallery.querySelector("[data-example-count]");
    const empty = gallery.querySelector("[data-example-empty]");
    const cards = Array.from(gallery.querySelectorAll("[data-example-card]"), (element) => ({
        element,
        goal: element.dataset.goal,
        feature: element.dataset.features.split("|"),
        setup: element.dataset.setup,
        text: element.textContent.toLowerCase(),
    }));
    const fields = form.elements;
    // Populate controls from the rendered catalog so new recipes need only metadata.
    for (const name of ["goal", "feature", "setup"]) {
        const values = new Set(cards.flatMap((card) => card[name]));
        for (const value of [...values].sort()) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = value;
            fields.namedItem(name).append(option);
        }
    }
    const update = () => {
        const terms = fields.namedItem("query").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
        const goal = fields.namedItem("goal").value;
        const feature = fields.namedItem("feature").value;
        const setup = fields.namedItem("setup").value;
        let visible = 0;
        for (const card of cards) {
            const matches = terms.every((term) => card.text.includes(term))
                && (!goal || card.goal === goal)
                && (!feature || card.feature.includes(feature))
                && (!setup || card.setup === setup);
            card.element.hidden = !matches;
            if (matches) visible += 1;
        }
        count.textContent = `${visible} of ${cards.length} starting points`;
        empty.hidden = visible !== 0;
    };
    form.addEventListener("submit", (event) => event.preventDefault());
    form.addEventListener("input", update);
    form.addEventListener("change", update);
    // Native reset updates field values after the event; refresh once it completes.
    form.addEventListener("reset", () => requestAnimationFrame(update));
    gallery.querySelector("[data-example-reset]").addEventListener("click", () => {
        form.reset();
        fields.namedItem("query").focus();
    });
    form.hidden = false;
    update();
})();
