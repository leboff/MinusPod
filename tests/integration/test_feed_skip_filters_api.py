"""Integration tests for per-podcast auto-process skip filters on PATCH /feeds/<slug>."""
import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

pytest.importorskip("ctranslate2", reason="Integration tests require Docker environment")


class TestSkipFiltersRoundtrip:
    def test_patch_persists_and_get_returns(self, temp_db, mock_podcast, app_client):
        slug = mock_podcast['slug']

        resp = app_client.patch(
            f'/api/v1/feeds/{slug}',
            data=json.dumps({'skipTitleRegex': '(?i)bonus', 'skipMaxDurationMinutes': 90}),
            content_type='application/json',
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['skipTitleRegex'] == '(?i)bonus'
        assert body['skipMaxDurationMinutes'] == 90

        get_resp = app_client.get(f'/api/v1/feeds/{slug}')
        assert get_resp.status_code == 200
        get_body = get_resp.get_json()
        assert get_body['skipTitleRegex'] == '(?i)bonus'
        assert get_body['skipMaxDurationMinutes'] == 90

    def test_invalid_regex_rejected(self, temp_db, mock_podcast, app_client):
        slug = mock_podcast['slug']
        resp = app_client.patch(
            f'/api/v1/feeds/{slug}',
            data=json.dumps({'skipTitleRegex': '[unterminated('}),
            content_type='application/json',
        )
        assert resp.status_code == 400

    def test_empty_and_zero_clear_filters(self, temp_db, mock_podcast, app_client):
        slug = mock_podcast['slug']
        app_client.patch(
            f'/api/v1/feeds/{slug}',
            data=json.dumps({'skipTitleRegex': 'x', 'skipMaxDurationMinutes': 5}),
            content_type='application/json',
        )
        resp = app_client.patch(
            f'/api/v1/feeds/{slug}',
            data=json.dumps({'skipTitleRegex': '', 'skipMaxDurationMinutes': 0}),
            content_type='application/json',
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['skipTitleRegex'] is None
        assert body['skipMaxDurationMinutes'] is None
