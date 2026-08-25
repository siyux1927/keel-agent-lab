"""AgentP - 一个面向生产的 Agent 运行时骨架。

四条主线:
  context/       上下文工程: 分层切片、Token 预算、压缩、组装
  memory/        记忆系统: 工作/情景/语义/程序性四层 + 混合检索 + 遗忘曲线
  loop/          Loop Engineering: ReAct 循环、预算护栏、死循环检测、熔断、反思重规划
  orchestrator/  多 Agent 编排: DAG 并发调度 + Planner/Worker/Critic
"""

__version__ = "0.1.0"
