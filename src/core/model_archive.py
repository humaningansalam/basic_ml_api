import stat
from lzma import LZMAError
from pathlib import Path, PurePosixPath
from zipfile import ZIP_BZIP2, BadZipFile, ZipFile, ZipInfo
from zlib import error as ZlibError

from src.common.errors import ApplicationError, ErrorCode

MAX_ARCHIVE_ENTRIES = 5_000


def extract_model_archive(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_uncompressed_bytes: int,
) -> None:
    """Validate and extract a model archive into an existing directory."""
    destination_directory = Path(destination)

    try:
        with ZipFile(archive_path, "r") as archive:
            entries = archive.infolist()
            _validate_archive_entries(
                entries,
                destination_directory,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
            if any(entry.compress_type == ZIP_BZIP2 for entry in entries):
                _verify_bzip2_entries(archive)
            archive.extractall(destination_directory)
    except ApplicationError:
        raise
    except (
        BadZipFile,
        EOFError,
        LZMAError,
        RuntimeError,
        UnicodeError,
        ZlibError,
    ) as error:
        raise ApplicationError(ErrorCode.INVALID_ZIP) from error


def _validate_archive_entries(
    entries: list[ZipInfo],
    destination_directory: Path,
    *,
    max_uncompressed_bytes: int,
) -> None:
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise ApplicationError(
            ErrorCode.ZIP_ENTRY_LIMIT_EXCEEDED,
            {
                "entry_count": len(entries),
                "max_entries": MAX_ARCHIVE_ENTRIES,
            },
        )

    destination_root = destination_directory.resolve()
    entry_kinds: dict[PurePosixPath, bool] = {}
    contains_model_artifact = False
    total_uncompressed_bytes = 0

    for entry in entries:
        entry_name = entry.filename
        filesystem_path = Path(entry_name)
        member_path = PurePosixPath(entry_name)
        is_directory = entry.is_dir()
        unix_file_mode = entry.external_attr >> 16
        is_symbolic_link = stat.S_ISLNK(unix_file_mode)
        has_windows_drive_prefix = len(entry_name) >= 2 and entry_name[1] == ":"
        is_absolute_path = (
            filesystem_path.is_absolute() or entry_name.startswith("\\") or has_windows_drive_prefix
        )

        if (
            is_absolute_path
            or is_symbolic_link
            or "\\" in entry_name
            or any(part == ".." for part in filesystem_path.parts)
        ):
            raise ApplicationError(ErrorCode.UNSAFE_ZIP_ENTRY, {"entry": entry_name})

        extraction_target = (destination_directory / filesystem_path).resolve()
        if not extraction_target.is_relative_to(destination_root):
            raise ApplicationError(ErrorCode.UNSAFE_ZIP_ENTRY, {"entry": entry_name})

        if member_path == PurePosixPath(".") and not is_directory:
            raise ApplicationError(ErrorCode.INVALID_ZIP)

        previous_kind = entry_kinds.get(member_path)
        if previous_kind is not None and previous_kind != is_directory:
            raise ApplicationError(ErrorCode.INVALID_ZIP)
        entry_kinds[member_path] = is_directory

        if not is_directory and entry_name.endswith(".keras"):
            contains_model_artifact = True

        total_uncompressed_bytes += entry.file_size

    for member_path in entry_kinds:
        for parent in member_path.parents:
            if parent == PurePosixPath("."):
                break
            if entry_kinds.get(parent) is False:
                raise ApplicationError(ErrorCode.INVALID_ZIP)

    if total_uncompressed_bytes > max_uncompressed_bytes:
        raise ApplicationError(
            ErrorCode.UNCOMPRESSED_SIZE_EXCEEDED,
            {
                "expanded_bytes": total_uncompressed_bytes,
                "max_bytes": max_uncompressed_bytes,
            },
        )

    if not contains_model_artifact:
        raise ApplicationError(ErrorCode.MODEL_ARTIFACT_REQUIRED)


def _verify_bzip2_entries(archive: ZipFile) -> None:
    # BZIP2 corruption can surface as OSError, which otherwise means storage failure.
    try:
        corrupt_entry_name = archive.testzip()
    except (
        BadZipFile,
        EOFError,
        LZMAError,
        OSError,
        RuntimeError,
        ZlibError,
    ) as error:
        raise ApplicationError(ErrorCode.INVALID_ZIP) from error

    if corrupt_entry_name is not None:
        raise ApplicationError(ErrorCode.INVALID_ZIP)
