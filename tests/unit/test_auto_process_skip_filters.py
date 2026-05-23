"""Tests for per-podcast auto-process skip filters (should_skip_auto_process)."""
import os
import sys
import tempfile

# Create temp data dir and set env before any imports that touch /app/data
_test_data_dir = tempfile.mkdtemp(prefix='skip_filter_test_')
os.environ.setdefault('SECRET_KEY', 'test-secret')
os.environ['DATA_DIR'] = _test_data_dir

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import database
import storage as storage_mod
database.Database._instance = None
database.Database.__init__.__defaults__ = (_test_data_dir,)
database.Database.__new__.__defaults__ = (_test_data_dir,)
storage_mod.Storage.__init__.__defaults__ = (_test_data_dir,)

from main_app.feeds import should_skip_auto_process


def _podcast(**overrides):
    base = {'skip_title_regex': None, 'skip_max_duration_minutes': None}
    base.update(overrides)
    return base


class TestTitleRegex:
    def test_match_anywhere_skips(self):
        podcast = _podcast(skip_title_regex='(?i)bonus')
        ep = {'title': 'Weekly Bonus Episode', 'duration': 600}
        skip, reason = should_skip_auto_process(podcast, ep)
        assert skip is True
        assert reason == 'title regex'

    def test_non_match_does_not_skip(self):
        podcast = _podcast(skip_title_regex='(?i)bonus')
        ep = {'title': 'Regular Episode 12', 'duration': 600}
        assert should_skip_auto_process(podcast, ep) == (False, None)

    def test_empty_regex_does_not_skip(self):
        podcast = _podcast(skip_title_regex='')
        ep = {'title': 'Bonus', 'duration': 600}
        assert should_skip_auto_process(podcast, ep) == (False, None)

    def test_missing_title_does_not_skip(self):
        podcast = _podcast(skip_title_regex='bonus')
        ep = {'duration': 600}
        assert should_skip_auto_process(podcast, ep) == (False, None)

    def test_invalid_regex_does_not_raise_or_skip(self):
        podcast = _podcast(skip_title_regex='[unterminated(')
        ep = {'title': 'anything', 'duration': 600}
        assert should_skip_auto_process(podcast, ep) == (False, None)


class TestMaxDuration:
    def test_over_limit_skips(self):
        podcast = _podcast(skip_max_duration_minutes=60)
        ep = {'title': 'Long one', 'duration': 3601}
        skip, reason = should_skip_auto_process(podcast, ep)
        assert skip is True
        assert reason == 'max duration'

    def test_under_limit_does_not_skip(self):
        podcast = _podcast(skip_max_duration_minutes=60)
        ep = {'title': 'Short one', 'duration': 3599}
        assert should_skip_auto_process(podcast, ep) == (False, None)

    def test_exactly_at_limit_does_not_skip(self):
        podcast = _podcast(skip_max_duration_minutes=60)
        ep = {'title': 'Exact', 'duration': 3600}
        assert should_skip_auto_process(podcast, ep) == (False, None)

    def test_unknown_duration_does_not_skip(self):
        podcast = _podcast(skip_max_duration_minutes=60)
        ep = {'title': 'Unknown length', 'duration': None}
        assert should_skip_auto_process(podcast, ep) == (False, None)


class TestNoFilters:
    def test_no_filters_set_does_not_skip(self):
        podcast = _podcast()
        ep = {'title': 'Anything', 'duration': 999999}
        assert should_skip_auto_process(podcast, ep) == (False, None)

    def test_title_filter_takes_precedence_reason(self):
        podcast = _podcast(skip_title_regex='(?i)bonus', skip_max_duration_minutes=1)
        ep = {'title': 'Bonus', 'duration': 999999}
        skip, reason = should_skip_auto_process(podcast, ep)
        assert skip is True
        assert reason == 'title regex'
