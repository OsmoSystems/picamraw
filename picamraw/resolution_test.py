from . import resolution as module


class TestPiResolution:
    def test_pad__adds_default_padding(self):
        actual = module.PiResolution(1920, 1080).pad()
        expected = module.PiResolution(width=1920, height=1088)

        assert actual == expected

    def test_pad__adds_custom_padding(self):
        actual = module.PiResolution(100, 100).pad(16, 16)
        expected = module.PiResolution(width=112, height=112)

        assert actual == expected
