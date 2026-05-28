const roleNames = {
  werewolf: "狼人",
  seer: "预言家",
  witch: "女巫",
  hunter: "猎人",
  villager: "村民",
};

const typeNames = {
  setup: "系统",
  phase: "阶段",
  private_action: "私密行动",
  speech: "发言",
  vote: "投票",
  death: "死亡",
  exile: "放逐",
  night_result: "夜晚结算",
  hunter_shot: "猎人开枪",
  game_over: "游戏结束",
};

let game = null;
let step = 0;

const els = {
  players: document.querySelector("#players"),
  winnerBadge: document.querySelector("#winnerBadge"),
  subtitle: document.querySelector("#subtitle"),
  currentEvent: document.querySelector("#currentEvent"),
  eventLog: document.querySelector("#eventLog"),
  metrics: document.querySelector("#metrics"),
  privateInfo: document.querySelector("#privateInfo"),
  timeline: document.querySelector("#timeline"),
  stepText: document.querySelector("#stepText"),
  loadDemo: document.querySelector("#loadDemo"),
  prev: document.querySelector("#prev"),
  next: document.querySelector("#next"),
};

async function loadDemo() {
  const response = await fetch("../logs/latest_game.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error("请先运行 python3 run_tournament.py --games 5 --seed 42 生成 logs/latest_game.json");
  }
  game = await response.json();
  step = game.events.length - 1;
  els.timeline.max = Math.max(game.events.length - 1, 0);
  render();
}

function render() {
  if (!game) {
    els.currentEvent.innerHTML = "<div class='eventType'>等待数据</div><p>点击“加载示例”读取最近一局结构化日志。</p>";
    return;
  }
  const visibleEvents = game.events.slice(0, step + 1);
  const dead = new Set(visibleEvents.filter((event) => event.type === "death").map((event) => Number(event.target)));
  els.stepText.textContent = `Step ${step + 1} / ${game.events.length}`;
  els.timeline.value = String(step);
  els.subtitle.textContent = `Seed ${game.seed} · ${game.reason || "对局进行中"}`;
  els.winnerBadge.textContent = game.winner === "wolves" ? "狼人胜利" : game.winner === "good" ? "好人胜利" : "进行中";
  els.players.innerHTML = Object.entries(game.roles)
    .map(([id, role]) => playerTemplate(Number(id), role, dead.has(Number(id))))
    .join("");
  renderCurrent(game.events[step]);
  renderLog(visibleEvents);
  renderMetrics(visibleEvents);
  renderPrivateInfo();
}

function playerTemplate(id, role, isDead) {
  const side = role === "werewolf" ? "wolves" : "good";
  return `
    <div class="player ${isDead ? "dead" : ""}">
      <div class="avatar">${id}</div>
      <div>
        <strong>${id} 号玩家</strong>
        <div class="role">${roleNames[role]}</div>
      </div>
      <span class="tag ${side}">${side === "wolves" ? "狼" : "好"}</span>
    </div>
  `;
}

function renderCurrent(event) {
  els.currentEvent.innerHTML = `
    <div class="eventType">Day ${event.day} · ${typeNames[event.type] || event.type}</div>
    <h2>${describeEvent(event)}</h2>
    <p>${event.text || extraLine(event)}</p>
  `;
}

function renderLog(events) {
  els.eventLog.innerHTML = events
    .map((event, idx) => `<div class="eventItem ${idx === step ? "active" : ""}">${idx + 1}. ${describeEvent(event)}</div>`)
    .join("");
}

function renderMetrics(events) {
  const votes = events.filter((event) => event.type === "vote").length;
  const wolfExiles = events.filter((event) => event.type === "exile" && event.role === "werewolf").length;
  const exiles = events.filter((event) => event.type === "exile").length;
  const deaths = events.filter((event) => event.type === "death").length;
  els.metrics.innerHTML = `
    <div class="metric"><strong>白天投票</strong>${votes} 次 Agent 决策</div>
    <div class="metric"><strong>放逐命中率</strong>${exiles ? Math.round((wolfExiles / exiles) * 100) : 0}%</div>
    <div class="metric"><strong>出局人数</strong>${deaths} 人</div>
    <div class="metric"><strong>当前天数</strong>第 ${events.at(-1)?.day ?? 0} 天</div>
  `;
}

function renderPrivateInfo() {
  const entries = Object.entries(game.private_events || {});
  els.privateInfo.innerHTML = entries
    .filter(([, events]) => events.length)
    .map(([id, events]) => {
      const lines = events.map((event) => {
        if (event.type === "wolf_team") return `狼队：${event.wolves.join("、")} 号`;
        if (event.type === "seer_result") return `验 ${event.target} 号：${event.side === "wolves" ? "狼人阵营" : "好人阵营"}`;
        return JSON.stringify(event);
      });
      return `<div class="secret"><strong>${id} 号私有视角</strong>${lines.join("<br />")}</div>`;
    })
    .join("");
}

function describeEvent(event) {
  if (event.type === "speech") return `${event.actor} 号发言：${event.text}`;
  if (event.type === "vote") return `${event.actor} 号投给 ${event.target} 号`;
  if (event.type === "death") return `${event.target} 号因 ${event.cause} 出局，身份为${roleNames[event.role]}`;
  if (event.type === "exile") return `${event.target} 号被放逐，身份为${roleNames[event.role]}`;
  if (event.type === "private_action") return `${event.actor} 执行 ${event.action}，目标 ${event.target}`;
  if (event.type === "night_result") return `夜晚死亡：${event.deaths.length ? event.deaths.join("、") : "无人死亡"}`;
  if (event.type === "game_over") return `游戏结束：${event.winner === "wolves" ? "狼人" : "好人"}胜利`;
  return event.text || `${typeNames[event.type] || event.type}`;
}

function extraLine(event) {
  return Object.entries(event)
    .filter(([key]) => !["type", "day", "actor", "text"].includes(key))
    .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(", ") : value}`)
    .join(" · ");
}

els.loadDemo.addEventListener("click", () => loadDemo().catch((err) => (els.currentEvent.textContent = err.message)));
els.prev.addEventListener("click", () => {
  if (!game) return;
  step = Math.max(0, step - 1);
  render();
});
els.next.addEventListener("click", () => {
  if (!game) return;
  step = Math.min(game.events.length - 1, step + 1);
  render();
});
els.timeline.addEventListener("input", (event) => {
  step = Number(event.target.value);
  render();
});

render();
