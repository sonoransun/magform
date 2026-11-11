/* magform docs: theme toggle, contents drawer, table wrapping, heading anchors, ledger filters. */
(function () {
  "use strict";
  var root = document.documentElement;

  /* ---- theme: follows the system until the reader chooses; the choice is remembered ---- */
  function currentTheme() {
    var set = root.getAttribute("data-theme");
    if (set === "dark" || set === "light") return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  var themeButton = document.getElementById("theme-toggle");
  if (themeButton) {
    themeButton.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("magform-theme", next); } catch (e) { /* private mode: keep for this page only */ }
      document.dispatchEvent(new CustomEvent("magform:theme", { detail: next }));
    });
  }

  /* ---- contents drawer on narrow screens ---- */
  var menuButton = document.getElementById("menu-toggle");
  if (menuButton) {
    var setOpen = function (open) {
      document.body.classList.toggle("nav-open", open);
      menuButton.setAttribute("aria-expanded", open ? "true" : "false");
      menuButton.setAttribute("aria-label", open ? "Close contents" : "Open contents");
    };
    menuButton.addEventListener("click", function () { setOpen(!document.body.classList.contains("nav-open")); });
    document.addEventListener("keydown", function (ev) { if (ev.key === "Escape") setOpen(false); });
    var rail = document.getElementById("rail");
    if (rail) rail.addEventListener("click", function (ev) { if (ev.target.closest("a")) setOpen(false); });
  }

  /* ---- wide tables scroll inside their own box, never the page ---- */
  Array.prototype.forEach.call(document.querySelectorAll(".prose table"), function (table) {
    if (table.closest(".table-scroll") || table.closest(".chart")) return;
    var box = document.createElement("div");
    box.className = "table-scroll";
    box.setAttribute("tabindex", "0");
    box.setAttribute("role", "region");
    box.setAttribute("aria-label", "Table");
    table.parentNode.insertBefore(box, table);
    box.appendChild(table);
  });

  /* ---- link anchors on section headings ---- */
  Array.prototype.forEach.call(document.querySelectorAll(".prose h2[id], .prose h3[id]"), function (h) {
    var a = document.createElement("a");
    a.className = "anchor";
    a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to this section");
    a.textContent = "#";
    h.appendChild(a);
  });

  /* ---- ledger: filter claim cards by verdict ---- */
  var filters = document.querySelector("[data-ledger-filters]");
  if (filters) {
    var cards = Array.prototype.slice.call(document.querySelectorAll(".ledger .claim"));
    var sections = Array.prototype.slice.call(document.querySelectorAll(".ledger__section"));
    var count = document.querySelector("[data-ledger-count]");
    var apply = function (verdict) {
      var shown = 0;
      cards.forEach(function (card) {
        var on = verdict === "all" || card.getAttribute("data-verdict") === verdict;
        card.hidden = !on;
        if (on) shown += 1;
      });
      sections.forEach(function (sec) {
        sec.hidden = !sec.querySelector(".claim:not([hidden])");
      });
      Array.prototype.forEach.call(filters.querySelectorAll(".chip"), function (chip) {
        chip.setAttribute("aria-pressed", chip.getAttribute("data-verdict") === verdict ? "true" : "false");
      });
      if (count) count.textContent = shown + " of " + cards.length + " shown";
    };
    filters.addEventListener("click", function (ev) {
      var chip = ev.target.closest(".chip");
      if (chip) apply(chip.getAttribute("data-verdict"));
    });
    apply("all");
  }
})();
