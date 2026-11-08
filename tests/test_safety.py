"""run_safely() must never let an exception message reach the console."""

from possession.safety import SHOW_ERRORS_ENV, run_safely


def test_message_is_withheld(capsys, monkeypatch):
    monkeypatch.delenv(SHOW_ERRORS_ENV, raising=False)

    def fails():
        raise ValueError("0.4521,0.3310,7.25")  # stands in for a quoted line of tracking

    assert run_safely(fails) == 1
    err = capsys.readouterr().err
    assert "ValueError" in err
    assert "0.4521" not in err


def test_exit_code_passes_through():
    assert run_safely(lambda: 0) == 0
    assert run_safely(lambda: None) == 0
    assert run_safely(lambda: 3) == 3
