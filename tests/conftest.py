"""测试夹具。

全部测试都跑在 Mock Provider 上: 无需 API Key、完全确定性、CI 里零成本。
这正是"Provider 抽象"这个设计带来的直接收益。
"""

from __future__ import annotations

import pytest

from agentp.llm import build_provider, reset_provider, set_provider
from agentp.memory import MemoryManager, reset_memory, set_memory
from agentp.tools import get_registry


@pytest.fixture(autouse=True)
def _isolate():
    """每个用例前后重置全局单例, 并把全局 Provider 钉死在 Mock 上。

    "钉死"这一步是必须的, 不是保险起见: 只 reset 的话, get_provider() 会照着 .env
    重建 —— 一旦本地把 AGENTP_PROVIDER 改成真实厂商, 没有显式注入 provider 的用例
    (以及 kb_search 这种走全局单例的工具)就会真的去打网络请求。后果是测试变慢、
    要花钱、而且断言结果取决于当天模型的心情。测试的确定性不能依赖一个配置文件。
    """
    reset_provider()
    reset_memory()
    provider = build_provider("mock")
    set_provider(provider)
    set_memory(MemoryManager(provider=provider))
    get_registry().reset_breakers()
    yield
    reset_provider()
    reset_memory()


@pytest.fixture
def provider():
    return build_provider("mock")


@pytest.fixture
def memory(provider):
    return MemoryManager(provider)


@pytest.fixture
def sample_doc() -> str:
    return """# Agent 系统设计

## 上下文工程
上下文窗口是最稀缺的资源。切片策略决定检索质量上限。定长切片会截断句子。
递归切片按分隔符优先级下切, 是通用默认值。

### 预算调度
默认窗口 8192 tokens, 输出预留 1024 tokens, 安全边界 5%。

```python
def allocate(zones, budget):
    return plan
```

## 记忆系统
记忆分为工作、情景、语义、程序性四层。遗忘曲线半衰期默认 72 小时。
被检索命中相当于复习一次, 会重置衰减计时。

## 循环工程
ReAct 循环需要预算护栏、死循环检测和熔断器。反思在连续失败两次后触发。
"""
