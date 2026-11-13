"""The "agent working" overlay: the vendored HALOFLUX animation (PolyForm Noncommercial 1.0.0)."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from optics_graph import server
from optics_graph.runtime import Runtime

VENDOR = Path(server.__file__).parent / "vendor" / "haloflux"
REQUIRED_NOTICES = (
    "Required Notice: Copyright (c) 2026 Ryder191 (RYDER CODE / Ms MAIKEBAIs CAT) - https://github.com/Ryder191/haloflux",
    "Required Notice: Noncommercial use only. Any commercial use is strictly prohibited. No commercial licenses are offered.",
)


@pytest.fixture
def client(tmp_path):
    with TestClient(server.create_app(Runtime(tmp_path))) as c:
        yield c


def test_haloflux_files_are_served_and_only_they_may_be_framed(client):
    page = client.get("/haloflux/")  # the optics-modified copy, original notice kept
    assert page.status_code == 200 and '<script src="wp.js">' in page.text and "window.OPTICS" in page.text
    assert "MODIFIED from HALOFLUX" in page.text and "HALOFLUX - Copyright (c) 2026 Ryder191" in page.text
    upstream = client.get("/haloflux/upstream.html")
    assert upstream.status_code == 200 and upstream.text == (VENDOR / "index.html").read_text("utf-8")
    assert "frame-ancestors 'self'" in page.headers["content-security-policy"]
    assert client.get("/haloflux/wp.js").headers["content-type"].startswith("text/javascript")
    assert all(n in client.get("/haloflux/LICENSE").text for n in REQUIRED_NOTICES)
    assert "frame-ancestors 'none'" in client.get("/").headers["content-security-policy"]


@pytest.mark.parametrize("name", ["NOTICE.txt", "../server.py", "..%2Fserver.py", "audio/sw421k.mp3", "index.html",
                                  "optics.html", "index.html/x"])
def test_haloflux_route_serves_nothing_outside_its_allowlist(client, name):
    assert client.get("/haloflux/" + name).status_code == 404


def test_haloflux_notices_and_credit_are_kept():
    notice = (VENDOR / "NOTICE.txt").read_text()
    assert all(n in notice for n in REQUIRED_NOTICES)
    assert "HALOFLUX - Copyright (c) 2026 Ryder191" in (VENDOR / "index.html").read_text()
    html = (Path(server.__file__).parent / "panel.html").read_text()
    assert "HALOFLUX by Ryder191 (RYDER CODE / Ms MAIKEBAIs CAT)" in html
    assert 'href="https://github.com/Ryder191/haloflux"' in html
    assert "Simulation only · no hardware moves" in html
    assert "music:{value:false}" in html  # the panel never plays the upstream background music


def test_the_optics_copy_shows_no_invented_measurements():
    page = (VENDOR / "optics.html").read_text("utf-8")
    for gone in ("METHODS · ", "CFL = |u|", "Σ ρ dV", "const v = fmtNum()",
                 "pick(['read', 'build', 'review', 'qa', 'sync', 'diff'])"):
        assert gone not in page, gone
    assert "1/f = 1/s + 1/s′" in page and "simulated workbench · no hardware moves" in page
    for method in ("setStatus", "setVocab", "setChecks", "setTempo", "pulse"):
        assert f"{method}(" in page
    html = (Path(server.__file__).parent / "panel.html").read_text()
    assert "Animation modified from" in html and "feedHaloflux(" in html
