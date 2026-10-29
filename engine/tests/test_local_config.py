from clip_engine.config import Settings


def test_local_clipping_mode_uses_local_providers_without_fallback_models():
    settings = Settings(_env_file=None, clipping_mode="local")

    assert settings.transcription_provider == "local"
    assert settings.transcription_model == "base"
    assert settings.get_planner_fallback_models() == []
    assert settings.get_layout_vision_fallback_models() == []
