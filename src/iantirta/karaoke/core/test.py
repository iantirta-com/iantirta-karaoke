from dataclasses import dataclass

@dataclass
class Test:
    _lyric: str | None = None

    @property
    def lyric(self):
        if self._lyric is None:
            self._lyric = 'Hello'
        return self._lyric

    @lyric.setter
    def lyric(self, value: str):
        self._lyric = value


if __name__ == "__main__":
    from pprint import pprint
    from rich import inspect as i
    a = Test()
    pprint(getattr(a, 'lyric'))