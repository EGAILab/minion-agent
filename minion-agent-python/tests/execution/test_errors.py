"""`EXEC-001`: the shared error vocabulary and `to_fs_error`'s `OSError`-mapping, matching pinned
Pi's own `toFileError` exactly."""

import errno

from minion_agent.execution.errors import FsErrorCode, to_fs_error


def test_file_not_found_maps_to_not_found() -> None:
    error = to_fs_error(FileNotFoundError(), "/x")
    assert error.code == FsErrorCode.NOT_FOUND
    assert error.path == "/x"
    assert error.cause is not None


def test_permission_error_maps_to_permission_denied() -> None:
    assert to_fs_error(PermissionError()).code == FsErrorCode.PERMISSION_DENIED


def test_not_a_directory_error_maps_to_not_directory() -> None:
    assert to_fs_error(NotADirectoryError()).code == FsErrorCode.NOT_DIRECTORY


def test_is_a_directory_error_maps_to_is_directory() -> None:
    assert to_fs_error(IsADirectoryError()).code == FsErrorCode.IS_DIRECTORY


def test_einval_maps_to_invalid() -> None:
    exc = OSError()
    exc.errno = errno.EINVAL
    assert to_fs_error(exc).code == FsErrorCode.INVALID


def test_unmapped_errno_maps_to_unknown() -> None:
    exc = OSError()
    exc.errno = errno.ENOTEMPTY
    assert to_fs_error(exc).code == FsErrorCode.UNKNOWN


def test_no_errno_maps_to_unknown() -> None:
    assert to_fs_error(OSError("plain")).code == FsErrorCode.UNKNOWN


def test_path_defaults_to_none() -> None:
    assert to_fs_error(OSError("plain")).path is None
