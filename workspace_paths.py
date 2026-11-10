import os
from pathlib import Path


def resolve_workspace_path(
    workspace: Path,
    raw_path: str,
    *,
    allow_root: bool = False,
) -> Path:
    """将相对路径转换为 Workspace 内的绝对路径。"""

    # 1. 检查路径是否为空
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("文件路径不能为空")

    relative_path = Path(raw_path)

    # 2. 拒绝绝对路径、带盘符路径和父目录穿越
    if (
        relative_path.anchor
        or ".." in relative_path.parts
        or (relative_path == Path(".") and not allow_root)
    ):
        raise ValueError("只允许 Workspace 内的相对文件路径")

    # 3. 拒绝 Windows 保留路径，例如 CON、NUL
    if os.name == "nt" and os.path.isreserved(raw_path):
        raise ValueError("不允许使用 Windows 保留路径")

    # 4. 检查 Workspace 根目录
    if workspace.is_symlink() or workspace.is_junction():
        raise ValueError("Workspace 根目录不能是链接")

    root = workspace.resolve(strict=True)

    if not root.is_dir():
        raise ValueError("Workspace 必须是一个目录")

    # 5. 拒绝路径中的符号链接和 Windows Junction
    current = root

    for part in relative_path.parts:
        current = current / part

        if current.is_symlink() or current.is_junction():
            raise ValueError("不允许通过链接访问文件")

    # 6. 解析最终路径，并检查是否仍处于 Workspace 内
    target = current.resolve(strict=False)

    if not target.is_relative_to(root):
        raise ValueError("文件路径超出了 Workspace 范围")

    return target
