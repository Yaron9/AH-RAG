import unittest

from ah_rag.utils.config import load_config


class TestConfigContracts(unittest.TestCase):
    def test_config_contains_core_sections(self) -> None:
        cfg = load_config()
        for section in ("search", "inference", "agent", "answer", "evaluation"):
            self.assertIn(section, cfg, msg=f"missing section: {section}")
        self.assertGreater(cfg.get("inference", {}).get("steps", 0), 0)
        self.assertIsInstance(cfg.get("answer", {}).get("use_llm"), bool)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
