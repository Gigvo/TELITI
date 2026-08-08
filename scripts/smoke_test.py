"""Post-deployment smoke test.

A deploy that goes green is not a deploy that works. The failure this exists to
catch is the quiet one: the container starts, serves the UI, answers /health, and
returns 503 from every analysis because the weights were never copied in. Nothing
about that looks broken until someone tries it.

Run it against the deployed URL before showing anyone:

    python scripts/smoke_test.py https://teliti-api.fly.dev

Exit code is 0 only if every check passes, so it can gate a release.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 90  # Generous: a cold start reloads 541 MB of weights.

# The concept paper's own §3.4 example. If this stops scoring Tinggi, the demo
# that the paper describes no longer works.
SCAM_TEXT = (
    "LOWONGAN KERJA URGENT!!! Dibutuhkan admin online, gaji Rp9.000.000 per bulan, "
    "tanpa pengalaman, tanpa wawancara, langsung kerja dari rumah. "
    "Wajib transfer biaya administrasi Rp250.000 ke rekening perusahaan sebagai "
    "tanda keseriusan. Kuota terbatas, hubungi WhatsApp sekarang juga!"
)

LEGIT_TEXT = (
    "PT Teknologi Nusantara membuka lowongan untuk posisi Backend Engineer. "
    "Kualifikasi: S1 Teknik Informatika atau setara, pengalaman minimal 2 tahun "
    "dengan Python dan PostgreSQL, memahami REST API dan version control. "
    "Penempatan di Jakarta Selatan, sistem kerja hybrid. "
    "Gaji Rp12.000.000 - Rp18.000.000 tergantung pengalaman. "
    "Kirim CV ke karir@teknologinusantara.co.id. Proses seleksi meliputi tes "
    "teknis dan dua tahap wawancara."
)

_passed = 0
_failed = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
    return condition


def call(base: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    url = f"{base.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:
            return exc.code, {}
    except Exception as exc:  # DNS, TLS, refused, timeout
        return 0, {"detail": str(exc)}


def main(base: str) -> int:
    print(f"\nTELITI smoke test -> {base}\n")

    # --- health ---------------------------------------------------------
    print("health")
    started = time.time()
    status, health = call(base, "/health")
    elapsed = time.time() - started

    if not check("reachable", status == 200, health.get("detail", f"HTTP {status}")):
        print("\nThe host is not answering. Nothing else can be tested.\n")
        return 1

    print(f"        (responded in {elapsed:.1f}s)")

    # This is the check that catches the model-less image.
    check(
        "model_loaded is true",
        health.get("model_loaded") is True,
        "Weights are missing from the image. The container is serving, but every "
        "analysis will return 503. Confirm artifacts/ was in the build context "
        "and not excluded by .dockerignore.",
    )
    check(
        "thresholds_loaded is true",
        health.get("thresholds_loaded") is True,
        "Run: python ml/fit_thresholds.py, then rebuild.",
    )
    check(
        "model_version is not a stub",
        not str(health.get("model_version", "")).startswith("stub"),
        f"model_version={health.get('model_version')!r}",
    )

    # --- scoring --------------------------------------------------------
    print("\nscoring")
    status, scam = call(base, "/api/v1/analyze", {"text": SCAM_TEXT})
    if not check("scam ad analysed", status == 200, scam.get("detail", f"HTTP {status}")):
        print("\nAnalysis is failing. Check the container logs.\n")
        return 1

    status, legit = call(base, "/api/v1/analyze", {"text": LEGIT_TEXT})
    check("legitimate ad analysed", status == 200, legit.get("detail", f"HTTP {status}"))

    check(
        "paper scenario scores Tinggi",
        scam.get("risk_label") == "Tinggi",
        f"got {scam.get('risk_label')!r} at score {scam.get('integrity_score')}",
    )
    check(
        "legitimate ad is not Tinggi",
        legit.get("risk_label") != "Tinggi",
        f"false positive: scored {legit.get('integrity_score')} — §3.6 calls this "
        f"the expensive error",
    )
    check(
        "scam scores below legitimate",
        scam.get("integrity_score", 100) < legit.get("integrity_score", 0),
        f"scam={scam.get('integrity_score')} legit={legit.get('integrity_score')}",
    )

    # Calibration: with the EMSCAD calibrator every Indonesian ad landed in 93-100.
    check(
        "deployment calibrator is in use",
        scam.get("integrity_score", 100) < 90,
        "Score suggests the EMSCAD calibrator shipped instead of "
        "calibrator_deployment.json — run python ml/fit_thresholds.py.",
    )

    # --- contract the frontend depends on -------------------------------
    print("\nresponse contract")
    check("disclaimer present", bool(scam.get("disclaimer")), "Ethics requirement, §3.6")
    check("privacy note present", bool(scam.get("privacy_note")))
    check("sentence evidence returned", bool(scam.get("sentence_evidence")))
    check(
        "evidence spans index the submitted text",
        all(
            SCAM_TEXT[e["span"]["start"] : e["span"]["end"]] == e["text"]
            for e in scam.get("sentence_evidence", [])
            if e.get("span")
        ),
        "Highlights will land on the wrong words in the UI.",
    )
    check(
        "evidence is real occlusion, not keywords",
        scam.get("sentence_evidence_approximate") is False,
    )

    latency = scam.get("latency_ms", 0)
    check("latency under 3s", latency < 3000, f"{latency}ms")

    # --- appeals --------------------------------------------------------
    print("\nappeals")
    status, report = call(
        base,
        "/api/v1/report",
        {
            "correction": "false_positive",
            "text": LEGIT_TEXT,
            "reported_score": legit.get("integrity_score", 0),
            "reported_label": legit.get("risk_label", "Sedang"),
            "request_id": legit.get("request_id", ""),
            "comment": "smoke test — safe to delete",
        },
    )
    check("report accepted", status == 200, report.get("detail", f"HTTP {status}"))
    check("report id returned", bool(report.get("report_id")))

    # --- summary --------------------------------------------------------
    print(f"\n{'-' * 58}")
    print(f"{_passed} passed, {_failed} failed")
    if _failed:
        print("\nNot ready to show. Fix the failures above.\n")
        return 1
    print("\nAll checks passed.")
    print("Still verify by hand: open the URL on a real phone, and submit a job")
    print("posting URL (outbound HTTP is blocked on some free tiers).\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: python scripts/smoke_test.py <base-url>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
