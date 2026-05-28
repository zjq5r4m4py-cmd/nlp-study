# AI 狼人杀 Agent Team

一个面向课程作业的多智能体协作与博弈系统。系统实现了狼人杀的信息隔离、多角色 Agent、完整回合引擎、胜负裁决、结构化日志、评测复盘与前端观战 UI。

## 项目亮点

- 多 Agent：狼人、预言家、女巫、猎人、村民均有独立目标、私有信息与行动策略。
- 信息隔离：每个 Agent 只能看到公开事件、自己的身份，以及角色允许获得的私有信息。
- 对局引擎：支持夜晚行动、白天发言、投票放逐、技能结算和胜负判定。
- 结构化日志：每局输出 `json` 日志，可用于可视化、复盘和评测。
- 进阶方向：选择「② 评测+复盘」，提供胜率、阵营表现、关键事件、投票一致性等指标。
- 前端观战 UI：可加载示例对局，按阶段查看 AI 发言、投票和行动轨迹。

## 快速运行

```bash
python3 run_tournament.py --games 5 --seed 42
python3 -m http.server 8000
```

然后打开：

```text
http://localhost:8000/web/
```

## 目录结构

```text
src/
  agents.py          # 各角色 Agent 策略
  engine.py          # 完整对局引擎
  evaluation.py      # 评测与复盘
run_tournament.py    # 批量运行入口
web/
  index.html         # 观战 UI
  styles.css
  app.js
logs/
  latest_game.json   # 最近一局结构化日志
  tournament.json    # 批量评测结果
```

## 可扩展方向

- 接入真实 LLM：将 `BaseAgent.speak` 和各类行动决策替换为 LLM 调用。
- 人机混战：在前端指定玩家为 human，由后端等待人工输入。
- 自进化 Agent：基于 `evaluation.py` 的复盘结果自动调整 Agent 权重参数。
