"""Real-browser check (slow): the dark-mode toggle changes the colours the browser actually paints, in the sidebar and in the main area,
and keeps them while moving between pages. Skipped when Playwright or its Chromium is not installed.

Optional: set RIVERDEMAND_SCREENSHOT_DIR to a folder to also save a screenshot of each page in each theme.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
playwright_sync = pytest.importorskip("playwright.sync_api", reason="playwright is not installed (pip install playwright; python -m playwright install chromium)")

ROOT = Path(__file__).resolve().parents[2]
PAGE_URLS = {"Overview": "", "Forecast Explorer": "Forecast_Explorer", "Model Comparison": "Model_Comparison", "Inventory Decisions": "Inventory_Decision",
             "Model Reliability": "Model_Reliability", "About & Methodology": "About_Methodology"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", str(port), "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(base + "/_stcore/health", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            pytest.skip("the Streamlit server did not start in time")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # pragma: no cover - browser binary missing
            pytest.skip(f"Chromium is not available: {exc}")
        yield b
        b.close()


def _rgb(text: str):
    return tuple(int(v) for v in re.findall(r"\d+", text)[:3])


def _luma(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _paint(page):
    return page.evaluate("""() => ({
        app: getComputedStyle(document.querySelector('[data-testid="stApp"]')).backgroundColor,
        side: getComputedStyle(document.querySelector('[data-testid="stSidebar"]')).backgroundColor,
        text: getComputedStyle(document.querySelector('h1')).color })""")


def _open(browser, url, size=(1366, 768)):
    page = browser.new_context(viewport={"width": size[0], "height": size[1]}).new_page()
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("h1", timeout=60000)
    page.wait_for_timeout(1500)
    return page


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_dark_and_light_paint_differently_in_the_browser(server, browser, size):
    light = _paint(_open(browser, server + "/?theme=light", size))
    dark = _paint(_open(browser, server + "/?theme=dark", size))
    assert _luma(_rgb(light["app"])) > 200 and _luma(_rgb(dark["app"])) < 60
    assert _luma(_rgb(light["side"])) > 180 and _luma(_rgb(dark["side"])) < 70, "the sidebar does not follow the theme"
    assert _luma(_rgb(light["text"])) < 90 and _luma(_rgb(dark["text"])) > 170, "headings keep their colour across themes"


def test_the_toggle_switches_the_live_page_and_the_choice_survives_navigation(server, browser):
    page = _open(browser, server + "/?theme=light")
    assert _luma(_rgb(_paint(page)["app"])) > 200
    page.get_by_text("Dark mode", exact=True).first.click()
    page.wait_for_timeout(2500)
    after = _paint(page)
    assert _luma(_rgb(after["app"])) < 60 and _luma(_rgb(after["side"])) < 70
    page.get_by_role("link", name=re.compile("Model Comparison")).first.click()
    page.wait_for_selector("h1:has-text('Model Comparison')", timeout=60000)
    page.wait_for_timeout(2000)
    assert _luma(_rgb(_paint(page)["app"])) < 60, "dark mode was lost when changing page"
    page.get_by_text("Dark mode", exact=True).first.click()
    page.wait_for_timeout(2500)
    assert _luma(_rgb(_paint(page)["app"])) > 200


@pytest.mark.parametrize("name,slug", list(PAGE_URLS.items()))
@pytest.mark.parametrize("theme_name", ["light", "dark"])
def test_each_page_loads_in_the_browser_with_no_script_error_and_no_sideways_scroll(server, browser, name, slug, theme_name):
    page = _open(browser, f"{server}/{slug}?theme={theme_name}")
    assert page.locator('[data-testid="stException"]').count() == 0, f"{name}: Streamlit reported an exception"
    assert page.locator("h1").first.inner_text().strip().lower().startswith(name.split(" &")[0].lower()[:8])
    overflow = page.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 2, f"{name}: the page scrolls sideways by {overflow}px"
    folder = os.environ.get("RIVERDEMAND_SCREENSHOT_DIR")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(folder) / f"{slug or 'Overview'}_{theme_name}_{page.viewport_size['width']}.png"), full_page=True)
