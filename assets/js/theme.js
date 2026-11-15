// Day and night: follows the system until the visitor picks one, then remembers the choice.
(function () {
  var root = document.documentElement;
  var button = document.querySelector(".theme-toggle");
  var system = window.matchMedia("(prefers-color-scheme: dark)");
  if (!button) return;

  function mode() {
    return root.getAttribute("data-theme") || (system.matches ? "dark" : "light");
  }
  function show() {
    var dark = mode() === "dark";
    button.setAttribute("data-mode", dark ? "dark" : "light");
    button.setAttribute("aria-label", dark ? "Switch to day mode" : "Switch to night mode");
    button.setAttribute("title", dark ? "Switch to day mode" : "Switch to night mode");
  }
  button.addEventListener("click", function () {
    var next = mode() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    show();
  });
  if (system.addEventListener) system.addEventListener("change", show);
  show();
})();
