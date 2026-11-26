// pEigen site behavior: theme toggle, mobile menu, code highlighting and copy buttons.
// Plain script, no build step. Pages work without it; it only adds conveniences.
(function () {
  "use strict";

  var THEME_KEY = "peigen-theme";

  function storedTheme() {
    try {
      return localStorage.getItem(THEME_KEY);
    } catch (e) {
      return null;
    }
  }

  function effectiveTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  var SUN =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var MOON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  function paintThemeButton(btn) {
    var dark = effectiveTheme() === "dark";
    btn.innerHTML = dark ? SUN : MOON;
    btn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
    btn.title = btn.getAttribute("aria-label");
  }

  function initTheme() {
    var btn = document.querySelector("[data-theme-toggle]");
    if (!btn) return;
    paintThemeButton(btn);
    btn.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem(THEME_KEY, next);
      } catch (e) {
        /* storage unavailable: the choice lasts for this page view */
      }
      paintThemeButton(btn);
      document.dispatchEvent(new CustomEvent("peigen:theme"));
    });
    if (!storedTheme() && window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
        paintThemeButton(btn);
        document.dispatchEvent(new CustomEvent("peigen:theme"));
      });
    }
  }

  function initMenu() {
    var btn = document.querySelector("[data-menu-toggle]");
    var links = document.getElementById("nav-links");
    if (!btn || !links) return;
    btn.addEventListener("click", function () {
      var open = links.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // ---------- syntax highlighting (Python and shell) ----------

  var PY_KEYWORDS = new Set(
    (
      "False None True and as assert async await break class continue def del elif else except finally for from " +
      "global if import in is lambda nonlocal not or pass raise return try while with yield"
    ).split(" ")
  );
  var PY_BUILTINS = new Set(
    "print range len abs min max sum zip enumerate float int str list dict tuple set isinstance sorted round".split(" ")
  );

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function span(cls, text) {
    return '<span class="' + cls + '">' + escapeHtml(text) + "</span>";
  }

  // One pass over the source with a sticky regex: comments, strings (incl. f/r prefixes and
  // triple quotes), numbers, identifiers. Everything else is copied through escaped.
  var PY_TOKEN =
    /(#[^\n]*)|((?:[rRbBfFuU]{1,2})?(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(\b(?:0[xX][0-9a-fA-F_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)j?\b)|([A-Za-z_][A-Za-z0-9_]*)/g;

  function highlightPython(src) {
    var out = "";
    var last = 0;
    var m;
    PY_TOKEN.lastIndex = 0;
    while ((m = PY_TOKEN.exec(src))) {
      out += escapeHtml(src.slice(last, m.index));
      if (m[1]) out += span("tok-com", m[1]);
      else if (m[2]) out += span("tok-str", m[2]);
      else if (m[3]) out += span("tok-num", m[3]);
      else {
        var word = m[4];
        var after = src.slice(PY_TOKEN.lastIndex).match(/^\s*\(/);
        if (PY_KEYWORDS.has(word)) out += span("tok-kw", word);
        else if (after || PY_BUILTINS.has(word)) out += span("tok-fn", word);
        else out += escapeHtml(word);
      }
      last = PY_TOKEN.lastIndex;
    }
    return out + escapeHtml(src.slice(last));
  }

  var SH_TOKEN = /(#[^\n]*)|("(?:\\.|[^"\\])*"|'[^']*')|(^|\n)(\s*)([A-Za-z0-9_.\-/]+)/g;

  function highlightShell(src) {
    var out = "";
    var last = 0;
    var m;
    SH_TOKEN.lastIndex = 0;
    while ((m = SH_TOKEN.exec(src))) {
      out += escapeHtml(src.slice(last, m.index));
      if (m[1]) out += span("tok-com", m[1]);
      else if (m[2]) out += span("tok-str", m[2]);
      else out += escapeHtml(m[3] + m[4]) + span("tok-fn", m[5]);
      last = SH_TOKEN.lastIndex;
    }
    return out + escapeHtml(src.slice(last));
  }

  function initCode() {
    var blocks = document.querySelectorAll("pre > code");
    blocks.forEach(function (code) {
      var pre = code.parentElement;
      var lang = (code.className.match(/language-(\w+)/) || [])[1];
      var src = code.textContent;
      if (lang === "python") code.innerHTML = highlightPython(src);
      else if (lang === "bash") code.innerHTML = highlightShell(src);

      if (pre.parentElement.classList.contains("code-block")) {
        addCopyButton(pre.parentElement, function () {
          return src;
        });
      }
    });
    document.querySelectorAll("[data-copy]").forEach(function (el) {
      addCopyButton(el, function () {
        return el.getAttribute("data-copy");
      });
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy") ? resolve() : reject(new Error("copy failed"));
      } catch (e) {
        reject(e);
      } finally {
        document.body.removeChild(ta);
      }
    });
  }

  function addCopyButton(container, getText) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy to clipboard");
    btn.addEventListener("click", function () {
      copyText(getText().replace(/\n$/, "")).then(
        function () {
          btn.textContent = "Copied";
          btn.classList.add("copied");
          setTimeout(function () {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 1600);
        },
        function () {
          btn.textContent = "Press ⌘C";
        }
      );
    });
    container.appendChild(btn);
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    initMenu();
    initCode();
  });
})();
