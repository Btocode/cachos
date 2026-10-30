import cachos.registry as R
from cachos import TTS, available_tts

from .conftest import FakeTTS


class _EP:
    def __init__(self, name, obj):
        self.name, self._obj = name, obj

    def load(self):
        return self._obj


class PluginTTS(FakeTTS):
    def __init__(self, **kw):
        super().__init__("plugtts", ["en", "sw"])


def test_entry_point_engine_class_and_register_hook(monkeypatch):
    def register():
        R.register_tts("hooktts", lambda **kw: FakeTTS("hooktts", ["en"]))

    monkeypatch.setattr(R, "entry_points", lambda group: [_EP("plugtts", PluginTTS), _EP("hook", register)])
    monkeypatch.setattr(R, "_plugins_loaded", False)
    monkeypatch.setattr(R, "_TTS", dict(R._TTS))
    assert {"plugtts", "hooktts"} <= set(available_tts())
    tts = TTS(["plugtts"])
    assert tts.pick_voice("sw").id == "plugtts:sw_f"
