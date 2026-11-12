// Add a documentation version picker to the top-right corner of every page.
//
// docs_site.py gives every page a script tag for this file, with a
// data-version attribute naming the page's folder and a data-versions
// attribute pointing at versions.json at the site root. We read that list
// when the reader opens the page, so we offer every published version.
(function () {
  const script = document.currentScript;
  const current = script.dataset.version;
  const versionsUrl = new URL(script.dataset.versions, location.href);
  const siteRoot = new URL(".", versionsUrl).pathname;

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
      "margin-right: 12px; padding: 2px 4px; border-radius: 4px;";

    // We don't list previews in versions.json, so add the one the reader
    // is on below the development docs.
    const entries = versions.slice();
    if (!entries.some((entry) => entry.name === current)) {
      entries.push({ name: current, label: current, url: prefix });
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

    // Put the picker beside "View page source" at the top right of the
    // page, or pin it to the corner of the window on pages without one.
    const aside = document.querySelector(".wy-breadcrumbs-aside");
    if (aside) {
      aside.insertBefore(select, aside.firstChild);
    } else {
      select.style.cssText +=
        "position: fixed; top: 8px; right: 12px; z-index: 300;";
      document.body.appendChild(select);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
