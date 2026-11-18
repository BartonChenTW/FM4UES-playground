// Theme toggle (remembers the choice when browser storage is available) and code highlighting.
(function () {
  var root = document.documentElement;
  try { var saved = localStorage.getItem("fm4ues-theme"); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.querySelector(".theme-toggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var dark = root.getAttribute("data-theme")
          ? root.getAttribute("data-theme") === "dark"
          : window.matchMedia("(prefers-color-scheme: dark)").matches;
        var next = dark ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try { localStorage.setItem("fm4ues-theme", next); } catch (e) {}
      });
    }
    if (window.hljs) window.hljs.highlightAll();
  });
})();
