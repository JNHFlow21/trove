# TROVE

TROVE 是仅支持 macOS 的本地私密能力运行时。它向外部 Agent 返回有界、带引用的
证据，并可选择运行本地 Reply Runtime，经由已验证的来源 Provider 生成、审核并
投递回复。回复投递默认关闭。MCP 是主要的 Agent 接口，CLI 是恢复与操作员接口。

产品是 **TROVE**。微信支持是一个可选的来源 Provider，而不是产品身份。

## 安装

在已验证的发布产物目录中：

```bash
python3 -m venv "$HOME/.local/share/trove/runtime"
"$HOME/.local/share/trove/runtime/bin/pip" install ./trove_runtime-1.0.0-py3-none-any.whl ./trove_provider_*.whl
export PATH="$HOME/.local/share/trove/runtime/bin:$PATH"
trove version
```

保持产物目录与 Vault 仅属主可访问。创建或选择一个 Vault，然后运行已脱敏的
健康检查。显式路径避免隐式发现。

```bash
export TROVE_VAULT_ROOT="$HOME/Trove/trove-vault"
mkdir -p "$TROVE_VAULT_ROOT"
chmod 700 "$TROVE_VAULT_ROOT"
trove --vault "$TROVE_VAULT_ROOT" doctor
```

## 连接 MCP

通过 Agent Switch 注册 `trove-mcp`，参数为：

```text
--pack standard --vault $TROVE_VAULT_ROOT
```

修改其中央配置前运行 `agent-switch doctor`，之后运行
`agent-switch reconcile`。不要手工编辑生成的客户端配置。日常回忆与搜索使用
standard pack 即可。

### 不使用 Agent Switch

直接在每个客户端中注册已安装的 `trove-mcp`（源码检出使用
`.venv/bin/trove-mcp`）。Claude Code：

```text
claude mcp add trove -- "$HOME/.local/share/trove/runtime/bin/trove-mcp" --pack standard --vault "$TROVE_VAULT_ROOT"
```

Codex（会写入 `~/.codex/config.toml` 的 `[mcp_servers.trove]`）：

```text
codex mcp add trove -- "$HOME/.local/share/trove/runtime/bin/trove-mcp" --pack standard --vault "$TROVE_VAULT_ROOT"
```

用 `bash scripts/install_skills.sh` 安装随仓 Skill：它把 `skills/*` 以符号链接
装入 `~/.agents/skills`，支持 `--target DIR` 指定目录，`--uninstall` 只移除这些
链接。

密钥取值只经由 Agent Switch 解析；环境变量用于启用 Provider 和选择密钥名，
绝不承载取值。没有 Agent Switch 时，本地 embedding 与全部 Vault 读取仍然可用；
云端能力与来源 Provider 密钥捕获需要 Agent Switch
（github.com/JNHFlow21/agent-switch）。

## 第一次调用

让 Agent 调用 `trove_recall`，或使用完全等价的 CLI 兜底：

```bash
trove --vault "$TROVE_VAULT_ROOT" recall --target "Example person" --limit 50
```

JSON 信封给出 `ok`、类型化错误、引用与覆盖度。仅当请求的覆盖度需要下一页时
才跟随不透明游标。

## 故障路径

运行 `trove --vault "$TROVE_VAULT_ROOT" doctor`。仅当 `error.retryable` 为真时
重试。遇到 `ambiguous_target` 时，从返回的账户中选定一个。遇到
`approval_required` 时停下：Agent 可以请求或查看审批，但只有控制终端前的人
才能决定。

参见 [MCP](docs/mcp.md)、[operations](docs/operations.md) 与生成的
[能力参考](docs/capability-map.md)。Provider 设置是独立的；参见
[已安装的来源 Provider](docs/providers/wechat.md)。可选 Reply Runtime 的架构与
安全模型见 [Reply Runtime](docs/architecture/reply-runtime.md)。

## 隐私边界

真实聊天数据库、导出、媒体、转录、Provider 载荷、密钥、日志与本地 Vault 数据
不属于本仓库。测试与入库证据必须是合成的或明确来源安全的。每次提交前运行
隐私扫描。

参见[开源隐私](PRIVACY.md)与[安全策略](SECURITY.md)。

## 贡献

参见 [CONTRIBUTING.md](CONTRIBUTING.md)。TROVE 采用
[Apache License 2.0](LICENSE) 许可。
