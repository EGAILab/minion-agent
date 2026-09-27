"""Mounting `read` and `ls` (WP-13.1) on the runtime, over the Runtime-mounted `ctx.fs`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...execution import FileSystem
from ...runtime import Context, plugin
from ..registry import register_tool
from .ls import create_ls_tool
from .read import ModelSupportsImages, ReadToolOptions, create_read_tool


class FsQueryToolsConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    auto_resize_images: bool = True
    """Pi's `autoResizeImages`."""
    model_supports_images: ModelSupportsImages | None = None
    """Whether the current request's model accepts images; see `read.ModelSupportsImages`."""


@plugin(name="builtin-fs-query-tools", inject=["fs", "tools"], config=FsQueryToolsConfig)
async def fs_query_tools_plugin(ctx: Context, config: FsQueryToolsConfig) -> None:
    """Register `read` and `ls` as reversible effects; both withdraw when this plugin unloads."""
    fs: FileSystem = ctx.fs
    options = ReadToolOptions(
        auto_resize_images=config.auto_resize_images,
        model_supports_images=config.model_supports_images,
    )
    register_tool(ctx, create_read_tool(fs, options))
    register_tool(ctx, create_ls_tool(fs))
