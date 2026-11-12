// Add a documentation version picker to the sidebar of every page.
//
// We publish this file at the site root, and give every page a script
// tag for it with a data-version attribute naming the page's folder. We
// read versions.json from the same folder as this script, so we list every
// published version when the reader opens the page.
(function () {
  const script = document.currentScript;
  const current = script.dataset.version;
  const siteRoot = new URL(".", script.src).pathname;
  const versionsUrl = new URL("versions.json", script.src);

  async function pageExists(url) {
    try {
      const response = await fetch(url, { method: "HEAD" });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  async function build() {
    let versions;
    try {
      const response = await fetch(versionsUrl);
      if (!response.ok) {
        return;
      }
      versions = await response.json();
    } catch (error) {
      return;
    }

    // Keep the reader on the same page when they switch version.
    const prefix = siteRoot + current + "/";
    const page = location.pathname.startsWith(prefix)
      ? location.pathname.slice(prefix.length)
      : "";

    const select = document.createElement("select");
    select.setAttribute("aria-label", "Documentation version");
    select.style.cssText =
      "margin-top: 8px; width: 100%; padding: 2px; border-radius: 4px;";

    // A pull-request preview is not listed, so show it as its own entry.
    const entries = versions.slice();
    if (!entries.some((entry) => entry.name === current)) {
      entries.unshift({ name: current, label: current, url: prefix });
    }
    for (const entry of entries) {
      const option = document.createElement("option");
      option.value = entry.url;
      option.textContent = entry.label;
      option.selected = entry.name === current;
      select.appendChild(option);
    }

    select.addEventListener("change", async () => {
      const target = select.value + page;
      location.href = (await pageExists(target)) ? target : select.value;
    });

    const anchor = document.querySelector(".wy-side-nav-search");
    if (anchor) {
      anchor.appendChild(select);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
