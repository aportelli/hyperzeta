from typing import Callable, Dict, List

import mlx.core as mx


class StreamArray:
    _streams: Dict[mx.DeviceType, List[mx.Stream]]

    def __init__(self) -> None:
        self._streams = {}

    def _extend(self, device: mx.DeviceType, size: int) -> None:
        streams = self._streams.setdefault(device, [])
        d = mx.Device(device)
        while len(streams) < size:
            streams.append(mx.new_stream(d))

    def eval(
        self, func: Callable[[int], mx.array], size: int, *, device: mx.DeviceType
    ) -> List[mx.array]:
        results = []
        self._extend(device, size)
        streams = self._streams[device][:size]
        for stream, i in zip(streams, range(size)):
            with mx.stream(stream):
                result = func(i)
                mx.async_eval(result)
                results.append(result)
        for stream in streams:
            mx.synchronize(stream)

        return results

    def chunked_reduction(
        self,
        func: Callable[[int, int], mx.array],
        size: int,
        *,
        n_threads: int = 1,
        device: mx.DeviceType,
    ) -> float:
        if size <= 0:
            return 0.0
        if n_threads <= 1:
            return func(0, size).item()
        n = min(size, n_threads)
        chunks = [(i * size // n, (i + 1) * size // n) for i in range(n)]
        partials = self.eval(lambda i: func(*chunks[i]), len(chunks), device=device)

        return sum(part.item() for part in partials)
