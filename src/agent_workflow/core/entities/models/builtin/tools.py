from __future__ import annotations

from typing import Any

from agent_workflow.core.entities.models.builtin.edit_file import (
    EditFileInput,
    edit_file,
)
from agent_workflow.core.entities.models.builtin.execute_shell import (
    ExecuteShellInput,
    execute_shell,
)
from agent_workflow.core.entities.models.builtin.find_files import (
    FindFilesInput,
    find_files,
)
from agent_workflow.core.entities.models.builtin.list_directory import (
    ListDirectoryInput,
    list_directory,
)
from agent_workflow.core.entities.models.builtin.read_file import (
    ReadFileInput,
    read_file,
)
from agent_workflow.core.entities.models.builtin.search_files import (
    SearchFilesInput,
    search_files,
)
from agent_workflow.core.entities.models.builtin.web_crawl import (
    WebCrawler,
    WebFetcher,
    create_web_crawl_tool,
)
from agent_workflow.core.entities.models.builtin.web_fetch import (
    WebFetchInput,
    web_fetch,
)
from agent_workflow.core.entities.models.builtin.write_file import (
    WriteFileInput,
    write_file,
)
from agent_workflow.core.entities.models.builtin.python_exec import (
    PythonExecInput,
    python_exec,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.graphql_query import (
    GraphQLQueryError,
    StashGraphQLClient,
    StashGraphQLInput,
    StashGraphQLPolicy,
    create_stash_graphql_tool,
    resolve_api_key,
)
from agent_workflow.core.entities.models.web_batch import (
    WebBatchFetcher,
    create_web_fetch_many_tool,
)
from agent_workflow.core.entities.models.web_research import (
    WebResearcher,
    create_web_research_tool,
)
from agent_workflow.core.entities.models.searxng_search import SearxngSearch
from agent_workflow.core.entities.models.web_search import (
    DuckDuckGoSearch,
    create_web_search_tool,
)
from agent_workflow.core.infrastructure.config import (
    GraphQLConfig,
    WebConfig,
    PythonConfig,
)


def _stash_client(
    config: GraphQLConfig,
):
    """Build the Stash client, or None when it is not configured."""

    endpoint = config.endpoint.strip()

    if not config.enabled or not endpoint:
        return None

    api_key = resolve_api_key(
        config.api_key,
        config.env_file,
    )

    if not api_key:
        return None

    return StashGraphQLClient(
        endpoint,
        api_key,
        policy=StashGraphQLPolicy(
            timeout=config.timeout,
            max_query_chars=config.max_query_chars,
            max_response_chars=config.max_response_chars,
            max_rows=config.max_rows,
            max_scan_rows=config.max_scan_rows,
            allow_introspection=config.allow_introspection,
        ),
    )


def create_stash_list_tags_tool(
    config: GraphQLConfig,
) -> Tool | None:
    """Build the compact id-and-name tag listing.

    None when Stash is unconfigured or has no key, so the tool is
    absent rather than present and failing.
    """

    client = _stash_client(config)

    if client is None:
        return None

    from agent_workflow.core.entities.models.stash_tags import (
        StashTagLister,
        TagListPolicy,
        create_stash_list_tags_tool as build,
    )

    return build(
        StashTagLister(
            client,
            policy=TagListPolicy(),
        )
    )


def create_stash_tag_overlap_tool(
    config: GraphQLConfig,
) -> Tool | None:
    """Build the usage-evidence tool for duplicate analysis.

    None when Stash is unconfigured or has no key, so the tool is
    absent rather than present and failing.
    """

    client = _stash_client(config)

    if client is None:
        return None

    from agent_workflow.core.entities.models.tag_overlap import (
        StashTagOverlap,
        TagOverlapPolicy,
        create_stash_tag_overlap_tool as build,
    )

    return build(
        StashTagOverlap(
            client,
            policy=TagOverlapPolicy(),
        )
    )


def create_stash_entity_schema_tool(
    config: GraphQLConfig,
) -> Tool | None:
    """Build the schema reader.

    None when Stash is unconfigured or has no key, so the tool is
    absent rather than present and failing.
    """

    client = _stash_client(config)

    if client is None:
        return None

    from agent_workflow.core.entities.models.stash_schema import (
        StashSchemaReader,
        create_stash_entity_schema_tool as build,
    )

    return build(StashSchemaReader(client))


def create_graphql_tool(
    config: GraphQLConfig,
) -> Tool | None:
    """Build the read-only GraphQL tool.

    Returns None when no endpoint is configured, so the tool is
    absent from the registry rather than present and failing. The
    tool can read the whole library, so being unable to reach one
    must not look like a permission problem at call time.
    """

    endpoint = config.endpoint.strip()

    if not config.enabled or not endpoint:
        return None

    api_key = resolve_api_key(
        config.api_key,
        config.env_file,
    )

    if not api_key:
        # Without a key the server answers 401, which reads like a
        # broken endpoint. Say what is actually missing.
        return Tool(
            name="stash_graphql",
            description=(
                "Read-only GraphQL access to the Stash database. "
                "Currently unavailable: no API key was found. Set "
                "graphql.api_key or point graphql.env_file at a "
                "file containing STASH_API_KEY."
            ),
            input_type=StashGraphQLInput,
            handler=_unconfigured_graphql,
            policy=ToolPolicy(
                permissions=frozenset({
                    "graphql.execute",
                }),
                timeout=10.0,
            ),
        )

    client = StashGraphQLClient(
        endpoint,
        api_key,
        policy=StashGraphQLPolicy(
            timeout=config.timeout,
            max_query_chars=config.max_query_chars,
            max_response_chars=config.max_response_chars,
            max_rows=config.max_rows,
            max_scan_rows=config.max_scan_rows,
            allow_introspection=config.allow_introspection,
        ),
    )

    return create_stash_graphql_tool(client)


async def _unconfigured_graphql(
    arguments: StashGraphQLInput,
    context: ToolContext,
) -> str:
    del arguments, context

    raise GraphQLQueryError(
        "stash_graphql is configured without an API key. Set "
        "graphql.api_key, or point graphql.env_file at a file "
        "containing STASH_API_KEY."
    )


def create_python_tool(
    config: PythonConfig,
) -> Tool:
    """Build the sandboxed python tool.

    The handler takes the configuration as a third parameter so the
    sandbox settings live in one place (``[python]`` in config.toml)
    rather than being frozen into the tool at import time.
    """

    async def handler(
        arguments: PythonExecInput,
        context: ToolContext,
    ) -> str:
        return await python_exec(
            arguments,
            context,
            config,
        )

    return Tool(
        name="python_exec",
        description=(
            "Run a Python script in a sandboxed directory and return "
            "its stdout, stderr and exit code. Use it for "
            "computation, data inspection, and writing throwaway "
            "scripts. The working directory is writable; scripts are "
            "subject to a timeout, a memory cap and an output cap. "
            "Imports of subprocess, socket, shutil and similar are "
            "refused, and the process runs as the agent's own user, "
            "so it can still read any file that user can read."
        ),
        input_type=PythonExecInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "python.execute",
            }),
            timeout=150.0,
            max_output_size=20_000,
        ),
    )


def build_search_backend(
    config: WebConfig | None = None,
) -> Any:
    """Pick the search engine from configuration.

    The default stays DuckDuckGo so an unconfigured install behaves as
    it did, but it is a scraper and will be challenged from most
    addresses; a SearXNG instance is the supported way to actually get
    results.
    """

    settings = config or WebConfig()

    if settings.backend == "searxng":
        from agent_workflow.core.entities.models.searxng_search import (
            SearxngPolicy,
            SearxngSearch,
        )

        return SearxngSearch(
            SearxngPolicy(
                endpoint=settings.searxng_endpoint,
                timeout=settings.searxng_timeout,
                categories=settings.searxng_categories,
            )
        )

    return DuckDuckGoSearch()


def create_builtin_tools(
    search: Any | None = None,
) -> tuple[Tool, ...]:
    # One HTTP fetcher behind every web tool. The crawler and the batch
    # fetcher both want a plain single-page WebFetcher; social expand
    # needs the same one, and passing the crawler instead silently broke
    # it, because the crawler only exposes crawl(), not fetch().
    web_fetcher = WebFetcher()

    web_crawler = WebCrawler(fetcher=web_fetcher)

    # Social expand reads link-in-bio and social pages, which serve an
    # empty shell to a plain HTTP request and are the entire reason the
    # tool exists. RenderFallbackFetcher tries the static fetch first
    # and only reaches for a browser when the page came back with
    # almost no links. Without a browser installed it degrades to the
    # static path rather than failing.
    from agent_workflow.core.entities.models.web_render import (
        BrowserFetcher,
        RenderFallbackFetcher,
    )

    expand_browser = BrowserFetcher()
    expand_fetcher: Any = RenderFallbackFetcher(
        web_fetcher,
        expand_browser if expand_browser.available else None,
    )

    web_crawl_tool = create_web_crawl_tool(
        web_crawler,
    )

    # One search backend and one batch fetcher shared by the three
    # web tools, instead of each tool building its own.
    web_search = search if search is not None else DuckDuckGoSearch()

    # The platform sweep needs a backend that can answer many queries;
    # the scraping default cannot, and the sweep is exactly the task a
    # model should not be left to loop over.
    from agent_workflow.core.entities.models.platform_scan import (
        create_web_platform_scan_tool,
    )

    platform_scan_tool = (
        create_web_platform_scan_tool(web_search)
        if isinstance(web_search, SearxngSearch)
        else None
    )

    web_batch_fetcher = WebBatchFetcher(
        fetcher=web_fetcher,
    )

    from agent_workflow.core.entities.models.social_expand import (
        create_web_social_expand_tool,
    )

    web_researcher = WebResearcher(
        web_search,
        web_batch_fetcher,
    )

    return (
        # --------------------------------------------------------------
        # Filesystem
        # --------------------------------------------------------------

        Tool(
            name="read_file",
            description="Read a UTF-8 text file",
            input_type=ReadFileInput,
            handler=read_file,
            policy=ToolPolicy(
                permissions=frozenset({
                    "filesystem.read",
                }),
                timeout=5.0,
                max_output_size=100_000,
            ),
        ),

        Tool(
            name="list_directory",
            description=(
                "List files and directories "
                "in a directory"
            ),
            input_type=ListDirectoryInput,
            handler=list_directory,
            policy=ToolPolicy(
                permissions=frozenset({
                    "filesystem.read",
                }),
                timeout=5.0,
                max_output_size=100_000,
            ),
        ),

        Tool(
            name="search_files",
            description=(
                "Search text content in files "
                "using a regular expression"
            ),
            input_type=SearchFilesInput,
            handler=search_files,
            policy=ToolPolicy(
                permissions=frozenset({
                    "filesystem.read",
                }),
                timeout=10.0,
                max_output_size=100_000,
            ),
        ),

        Tool(
            name="find_files",
            description=(
                "Find files by name pattern"
            ),
            input_type=FindFilesInput,
            handler=find_files,
            policy=ToolPolicy(
                permissions=frozenset({
                    "filesystem.read",
                }),
                timeout=10.0,
                max_output_size=100_000,
            ),
        ),

        Tool(
            name="write_file",
            description=(
                "Write UTF-8 text content "
                "to a file"
            ),
            input_type=WriteFileInput,
            handler=write_file,
            policy=ToolPolicy(
                permissions=frozenset({
                    "filesystem.write",
                }),
                timeout=5.0,
                max_output_size=1_000,
            ),
        ),

        Tool(
            name="edit_file",
            description=(
                "Replace exactly one occurrence "
                "of text in a UTF-8 text file"
            ),
            input_type=EditFileInput,
            handler=edit_file,
            policy=ToolPolicy(
                permissions=frozenset({
                    "filesystem.write",
                }),
                timeout=5.0,
                max_output_size=1_000,
            ),
        ),

        # --------------------------------------------------------------
        # Shell
        # --------------------------------------------------------------

        Tool(
            name="execute_shell",
            description=(
                "Execute a restricted shell command "
                "in the working directory. "
                "Only approved commands are allowed. "
                "Network access and other elevated "
                "capabilities may require explicit "
                "user approval."
            ),
            input_type=ExecuteShellInput,
            handler=execute_shell,
            policy=ToolPolicy(
                permissions=frozenset({
                    "shell.execute",
                }),
                timeout=15.0,
                max_output_size=20_000,
            ),
        ),

        # --------------------------------------------------------------
        # Web research
        # --------------------------------------------------------------

        Tool(
            name="web_fetch",
            description=(
                "Fetch one public HTTP or HTTPS "
                "web page and return structured "
                "research evidence containing "
                "the page content and discovered links."
            ),
            input_type=WebFetchInput,
            handler=web_fetch,
            policy=ToolPolicy(
                permissions=frozenset({
                    "web.fetch",
                }),
                timeout=20.0,
                max_output_size=120_000,
            ),
        ),

        web_crawl_tool,
        create_web_fetch_many_tool(
            web_batch_fetcher,
        ),
        create_web_search_tool(
            web_search,
        ),
        *(
            [platform_scan_tool]
            if platform_scan_tool is not None
            else []
        ),
        create_web_research_tool(
            web_researcher,
        ),
        create_web_social_expand_tool(
            expand_fetcher,
        ),
    )