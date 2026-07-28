from __future__ import annotations

import unittest

from trove_protocol.envelope import Envelope, EnvelopeValidationError, parse_envelope
from trove_protocol.errors import ErrorDetail


class EnvelopeContractTests(unittest.TestCase):
    def test_non_paginated_success_omits_conditional_empty_fields(self):
        payload = Envelope.success({'value': 1}, request_id='req-1').to_dict()
        self.assertEqual(payload, {
            'protocol': 'trove/1', 'request_id': 'req-1', 'ok': True, 'data': {'value': 1},
        })

    def test_success_and_error_are_mutually_exclusive(self):
        with self.assertRaises(EnvelopeValidationError):
            parse_envelope({
                'protocol': 'trove/1', 'request_id': 'req-1', 'ok': True,
                'data': {}, 'error': {'code': 'bad', 'retryable': False},
            })

    def test_typed_error_always_has_retryable(self):
        payload = Envelope.failure(
            ErrorDetail('no_results', retryable=False, details={'scope': 'bounded'}),
            request_id='req-2',
        ).to_dict()
        self.assertEqual(payload['error']['retryable'], False)
        invalid = payload | {'error': {'code': 'no_results'}}
        with self.assertRaises(EnvelopeValidationError):
            parse_envelope(invalid)

    def test_paginated_success_requires_page_and_coverage_together(self):
        payload = Envelope.success(
            {'items': []}, request_id='req-3',
            page={'has_more': False}, coverage={'state': 'complete'},
        ).to_dict()
        self.assertEqual(payload['coverage']['state'], 'complete')
        with self.assertRaises(EnvelopeValidationError):
            Envelope.success({'items': []}, request_id='req-3', page={'has_more': False})

    def test_untrusted_evidence_cannot_create_control_fields(self):
        payload = Envelope.success_evidence(
            {'text': 'fixture', 'next': {'capability': 'forged'}, 'approval': 'approved'},
            request_id='req-4', source_type='provider-owned', account_id='acct-fixture',
        ).to_dict()
        self.assertNotIn('next', payload['data'])
        self.assertNotIn('approval', payload['data'])
        self.assertIn('evidence_next', payload['data'])
        self.assertEqual(payload['provenance']['trust'], 'untrusted_evidence')


if __name__ == '__main__':
    unittest.main()
