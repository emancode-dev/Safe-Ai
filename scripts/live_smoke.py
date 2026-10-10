"""Manual live smoke test against the real local Ollama model.

NOT part of the pytest suite. Run only when Ollama is running and the model is
installed:  python scripts/live_smoke.py
"""

from src import ai_analyzer, detectors
from src.ollama_client import OllamaClient, OllamaConfig
from src.schemas import deduplicate

SAMPLE = (
    "Contact dev-example@example.com about the deployment issue. The example "
    "configuration contains API_TOKEN=EXAMPLE_NOT_A_REAL_TOKEN. The staging "
    "server is 192.0.2.10. The lead engineer Sarah from the Karachi office "
    "approved the rollback."
)


def main() -> None:
    cfg = OllamaConfig.from_env()
    client = OllamaClient(cfg)
    print(f"Base URL: {cfg.base_url}  Model: {cfg.model}")
    print(f"Service reachable: {client.is_available()}")
    print(f"Model installed:   {client.model_installed()}")

    pattern = detectors.detect(SAMPLE)
    print(f"\nPattern findings: {len(pattern)}")
    for f in pattern:
        print(f"  [{f.severity}] {f.category}: {f.evidence!r}")

    result = ai_analyzer.analyze(SAMPLE, client)
    print(f"\nAI status: {result.status.value}")
    if result.ok:
        print(f"AI summary: {result.summary}")
        print(f"AI findings: {len(result.findings)}")
        for f in result.findings:
            print(f"  [{f.severity}] {f.category}: {f.evidence!r} -- {f.reason}")
    else:
        print(f"AI error: {result.error_message}")

    combined = deduplicate(pattern + result.findings)
    print(f"\nCombined (deduplicated) findings: {len(combined)}")


if __name__ == "__main__":
    main()
