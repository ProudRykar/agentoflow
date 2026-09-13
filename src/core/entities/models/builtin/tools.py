
from core.entities.models.builtin.list_directory import (
    ListDirectoryInput,
    list_directory,
)
from core.entities.models.builtin.read_file import (
    ReadFileInput,
    read_file,
)
from core.entities.models.tool import Tool, ToolPolicy
from core.entities.models.builtin.search_files import (
    SearchFilesInput,
    search_files,
)
from core.entities.models.builtin.find_files import (
    FindFilesInput,
    find_files,
)
from core.entities.models.builtin.write_file import (
    WriteFileInput,
    write_file,
)
from core.entities.models.builtin.edit_file import (
    EditFileInput,
    edit_file,
)
def create_builtin_tools() -> tuple[Tool, ...]:
    return (
        Tool(
            name="read_file",
            description="Read a UTF-8 text file",
            input_type=ReadFileInput,
            handler=read_file,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=5.0,
                max_output_size=100_000,
            ),
        ),
        Tool(
            name="list_directory",
            description="List files and directories in a directory",
            input_type=ListDirectoryInput,
            handler=list_directory,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=5.0,
                max_output_size=100_000,
            ),
        ),
        Tool(
            name="search_files",
            description="Search text content in files using a regular expression",
            input_type=SearchFilesInput,
            handler=search_files,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=10.0,
                max_output_size=100_000,
            ),
        ),
        Tool(
            name="find_files",
            description="Find files by name pattern",
            input_type=FindFilesInput,
            handler=find_files,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=10.0,
                max_output_size=100_000,
            ),
        ),
        Tool(
            name="write_file",
            description="Write UTF-8 text content to a file",
            input_type=WriteFileInput,
            handler=write_file,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.write"}),
                timeout=5.0,
                max_output_size=1_000,
            ),
        ),
        Tool(
            name="edit_file",
            description="Replace exactly one occurrence of text in a UTF-8 text file",
            input_type=EditFileInput,
            handler=edit_file,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.write"}),
                timeout=5.0,
                max_output_size=1_000,
            ),
        ),
    )
