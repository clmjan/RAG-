from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deployment_manifests_and_env_template_are_present():
    assert (ROOT / "Dockerfile").is_file()
    assert (ROOT / "docker-compose.yml").is_file()
    assert (ROOT / ".env.example").is_file()
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "npm ci" in dockerfile and '"app.main:app"' in dockerfile
    assert "rag_data" in compose and "healthcheck:" in compose
    assert "HYBRID_LEXICAL_WEIGHT" in env and "LLM_API_BASE" in env
    assert "pytest -q" in workflow and "npm run build" in workflow
