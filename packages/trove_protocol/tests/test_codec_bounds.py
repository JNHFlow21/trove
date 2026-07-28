from __future__ import annotations

import struct
import time
import unittest

from trove_protocol.codec import FrameDecoder, decode_request, encode_frame
from trove_protocol.errors import ProtocolError


class CodecBoundsTests(unittest.TestCase):
    def _request(self) -> dict:
        return {
            'protocol': 'trove/1', 'request_id': 'req-1',
            'capability': 'trove.recall',
            'input': {'conversation_id': 'fixture', 'limit': 5},
            'deadline_ms': int(time.time() * 1000) + 10_000,
            'response_budget': 65_536,
        }

    def test_partial_frames_buffer_until_complete(self):
        request = self._request()
        frame = encode_frame(request)
        decoder = FrameDecoder()
        self.assertEqual(decoder.feed(frame[:3]), [])
        self.assertEqual(decoder.feed(frame[3:-1]), [])
        self.assertEqual(decoder.feed(frame[-1:]), [request])
        decoder.finish()

    def test_partial_frame_at_eof_fails_typed(self):
        decoder = FrameDecoder()
        decoder.feed(encode_frame(self._request())[:-1])
        with self.assertRaises(ProtocolError) as raised:
            decoder.finish()
        self.assertEqual(raised.exception.code, 'partial_frame')

    def test_oversize_and_unknown_fields_fail_closed(self):
        decoder = FrameDecoder(max_frame_bytes=8)
        with self.assertRaises(ProtocolError) as raised:
            decoder.feed(struct.pack('>I', 9))
        self.assertEqual(raised.exception.code, 'frame_too_large')
        request = self._request() | {'forged': True}
        with self.assertRaises(ProtocolError) as raised:
            decode_request(request)
        self.assertEqual(raised.exception.code, 'invalid_request')

    def test_protocol_capability_and_deadline_errors_are_typed(self):
        cases = (
            (self._request() | {'protocol': 'trove/2'}, 'protocol_mismatch'),
            (self._request() | {'capability': 'trove.unknown'}, 'unknown_capability'),
            (self._request() | {'deadline_ms': 1}, 'deadline_expired'),
        )
        for request, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ProtocolError) as raised:
                    decode_request(request)
                self.assertEqual(raised.exception.code, code)


if __name__ == '__main__':
    unittest.main()
