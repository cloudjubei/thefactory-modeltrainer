"""An exact arithmetic coder (Witten, Neal and Cleary, 1987) over adaptive symbols, for files whose every bit is proven
by reading them back. `Encoder` and `Decoder` take the same calls: `symbol` codes one of n values under a context whose
counts start at ALPHA_NUM/ALPHA_DEN each and learn as it goes, `uniform` one of n values equally likely, and a call
with n == 1 costs nothing. A decoder needs the bytes and the same sequence of calls, nothing else (it ignores a value
passed to it, so one walk can drive both), and reads zeros past the end of the file, so the encoder drops its trailing
zero bits. `Encoder.parts` adds up each family's ideal bits."""
from __future__ import annotations

import math

PRECISION = 32
TOP = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
ALPHA_NUM, ALPHA_DEN = 3, 10


class _Counts:
    def __init__(self):
        self.tables: dict = {}

    def table(self, family: str, context, n: int) -> list:
        return self.tables.setdefault((family, context, n), [ALPHA_NUM] * n)

    @staticmethod
    def learn(table: list, value: int) -> None:
        table[value] += ALPHA_DEN


class Encoder(_Counts):
    def __init__(self):
        super().__init__()
        self.low, self.high, self.pending, self.out = 0, TOP, 0, []
        self.parts: dict = {}

    def _emit(self, bit: int) -> None:
        self.out.append(bit)
        self.out.extend([1 - bit] * self.pending)
        self.pending = 0

    def _code(self, family: str, start: int, end: int, total: int) -> None:
        self.parts[family] = self.parts.get(family, 0.0) - math.log2((end - start) / total)
        span = self.high - self.low + 1
        self.high = self.low + span * end // total - 1
        self.low = self.low + span * start // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                return
            self.low *= 2
            self.high = self.high * 2 + 1

    def symbol(self, family: str, context, n: int, value: int) -> int:
        if not 0 <= value < n:
            raise ValueError(f"{value} is not one of {n} values")
        if n > 1:
            table = self.table(family, context, n)
            start = sum(table[:value])
            self._code(family, start, start + table[value], sum(table))
            self.learn(table, value)
        return value

    def uniform(self, family: str, n: int, value: int) -> int:
        if not 0 <= value < n:
            raise ValueError(f"{value} is not one of {n} values")
        if n > 1:
            self._code(family, value, value + 1, n)
        return value

    def finish(self) -> tuple:
        """(the file's bytes, its length in bits): the interval's closing bits, then no trailing zeros."""
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        bits = list(self.out)
        while bits and bits[-1] == 0:
            bits.pop()
        data = bytearray((len(bits) + 7) // 8)
        for i, bit in enumerate(bits):
            data[i // 8] |= bit << (7 - i % 8)
        return bytes(data), len(bits)


class Decoder(_Counts):
    def __init__(self, data: bytes):
        super().__init__()
        self.data, self.at = data, 0
        self.low, self.high, self.value = 0, TOP, 0
        for _ in range(PRECISION):
            self.value = self.value * 2 + self._bit()

    def _bit(self) -> int:
        i, self.at = self.at, self.at + 1
        return (self.data[i // 8] >> (7 - i % 8)) & 1 if i // 8 < len(self.data) else 0

    def _target(self, total: int) -> int:
        return ((self.value - self.low + 1) * total - 1) // (self.high - self.low + 1)

    def _take(self, start: int, end: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + span * end // total - 1
        self.low = self.low + span * start // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.value -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                return
            self.low *= 2
            self.high = self.high * 2 + 1
            self.value = self.value * 2 + self._bit()

    def symbol(self, family: str, context, n: int, _value=None) -> int:
        if n == 1:
            return 0
        table = self.table(family, context, n)
        total = sum(table)
        target = self._target(total)
        start = value = 0
        while start + table[value] <= target:
            start += table[value]
            value += 1
        self._take(start, start + table[value], total)
        self.learn(table, value)
        return value

    def uniform(self, family: str, n: int, _value=None) -> int:
        if n == 1:
            return 0
        value = self._target(n)
        self._take(value, value + 1, n)
        return value
