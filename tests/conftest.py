"""测试夹具。

全部测试都跑在 Mock Provider 上: 无需 API Key、完全确定性、CI 里零成本。
这正是"Provider 抽象"这个设计带来的直接收益。
"""

from __future__ import annotations

import pytest

from agentp.llm import build_provider, reset_provider
from agentp.memory import MemoryManager, reset_memory
from agentp.tools import get_registry


@pytest.fixture(autouse=True)
def _isolate():
    """每个用例前后重置全局单例, 避免记忆和熔断器状态互相串味。"""
    reset_provider()
    reset_memory()
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
