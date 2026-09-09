const TYPES = {
  forehand_clear: "正手高远",
  smash: "杀球",
  drop: "吊球",
  backhand_clear: "反手高远",
};

const CONNECTIONS = [
  [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 12], [23, 24], [11, 23], [12, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
];

function $(sel) { return document.querySelector(sel); }

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

function jobIdFromHash() {
  const m = location.hash.match(/^#\/jobs\/([^/]+)/);
  return m ? m[1] : null;
}

function renderHome(jobs) {
  document.getElementById("app").innerHTML = `
    <div class="page">
      <h1>羽毛球姿势教练</h1>
      <p class="muted">本机分析侧面后场练习视频。只需一块显示器。</p>
      <div class="card">
        <h2>新建分析</h2>
        <form id="new-job">
          <label>练习视频</label>
          <input name="video" type="file" accept="video/*" required />
          <label>整段动作类型</label>
          <select name="stroke_type">
            ${Object.entries(TYPES).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}
          </select>
          <label>持拍手</label>
          <select name="handedness">
            <option value="right" selected>右手</option>
            <option value="left">左手</option>
          </select>
          <label>机位</label>
          <input value="侧面（第一期锁定）" disabled />
          <label>示范视频（可选）</label>
          <input name="demo" type="file" accept="video/*" />
          <p><button type="submit">开始分析</button></p>
        </form>
      </div>
      <div class="card">
        <h2>历史任务</h2>
        ${jobs.length ? jobs.map(j => `
          <p><a href="#/jobs/${j.job_id}">${j.job_id}</a>
            · ${TYPES[j.stroke_type] || j.stroke_type}
            · ${j.status}</p>`).join("") : "<p class='muted'>暂无任务</p>"}
      </div>
    </div>`;
  $("#new-job").onsubmit = async (ev) => {
    ev.preventDefault();
    const fd = new FormData(ev.target);
    const res = await fetch("/jobs", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) {
      alert(body.detail || "上传失败");
      return;
    }
    location.hash = `#/jobs/${body.job_id}`;
  };
}

function drawPose(canvas, video, people) {
  const ctx = canvas.getContext("2d");
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!people || !people.length) return;
  const person = people[0];
  const lms = person.landmarks;
  ctx.strokeStyle = "#3dd68c";
  ctx.fillStyle = "#e8eef5";
  ctx.lineWidth = 2;
  for (const [a, b] of CONNECTIONS) {
    if (!lms[a] || !lms[b]) continue;
    ctx.beginPath();
    ctx.moveTo(lms[a].x * canvas.width, lms[a].y * canvas.height);
    ctx.lineTo(lms[b].x * canvas.width, lms[b].y * canvas.height);
    ctx.stroke();
  }
  for (const p of lms) {
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderJob(job) {
  const strokes = job.strokes || [];
  const selected = window._sel || (strokes[0] && strokes[0].stroke.stroke_id);
  const cur = strokes.find(s => s.stroke.stroke_id === selected) || strokes[0];
  const people = [...new Set(strokes.map(s => s.stroke.person_id))];
  document.getElementById("app").innerHTML = `
    <div class="page">
      <p><a href="#/">← 返回</a> · 任务 ${job.job_id} · ${job.status}</p>
      ${job.status === "error" ? `<p class="error">${job.error_message || "分析失败"}</p>
        <button id="retry">重试</button>` : ""}
      ${["queued","extracting_pose","segmenting","scoring"].includes(job.status)
        ? `<p class="muted">正在分析：${job.status} …</p>` : ""}
      <div class="layout">
        <div class="card">
          <h3>击球列表</h3>
          ${strokes.map(s => `
            <div class="stroke-item ${s.stroke.stroke_id === (cur && cur.stroke.stroke_id) ? "active" : ""} ${s.stroke.segment_confidence < 0.5 ? "low" : ""}"
                 data-id="${s.stroke.stroke_id}">
              ${s.stroke.start_s.toFixed(1)}s · ${TYPES[s.stroke.stroke_type]}
              · ${s.score.total.toFixed(0)}分
              ${s.stroke.segment_confidence < 0.5 ? "<div class='muted'>低置信度</div>" : ""}
            </div>`).join("") || "<p class='muted'>没有切出击球。请检查持拍手，或确认是侧面挥拍。</p>"}
        </div>
        <div class="card">
          <div class="player-wrap">
            <video id="v" src="/jobs/${job.job_id}/video" controls></video>
            <canvas id="c"></canvas>
          </div>
          <p>
            <button class="secondary" id="loop">只看这一击</button>
            <label>分析对象
              <select id="person">${people.map(id => `<option value="${id}" ${String(id)===String(job.person_id)?"selected":""}>${id}</option>`).join("")}</select>
            </label>
          </p>
          ${job.demo_error ? `<p class="muted">示范视频未使用：${job.demo_error}</p>` : ""}
          <div id="demo-slot"></div>
        </div>
        <div class="card" id="detail"></div>
      </div>
    </div>`;

  const detail = $("#detail");
  if (cur) {
    detail.innerHTML = `
      <h3>${TYPES[cur.stroke.stroke_type]} · ${cur.score.total.toFixed(0)}分</h3>
      <label>改类型</label>
      <select id="type">${Object.entries(TYPES).map(([k,v]) =>
        `<option value="${k}" ${k===cur.stroke.stroke_type?"selected":""}>${v}</option>`).join("")}</select>
      ${cur.score.metrics.map(m => `
        <div class="muted">${m.name} ${m.value.toFixed(1)}（${m.score.toFixed(0)}）</div>
        <div class="bar"><span style="width:${m.score}%"></span></div>`).join("")}
      <h3>建议</h3>
      ${(cur.score.advice || []).map(a => `<p>${a}</p>`).join("") || "<p class='muted'>暂无建议</p>"}
      ${(cur.demo_diffs || []).map(d =>
        `<p class="muted">示范对比 ${d.metric}：练习 ${d.practice_value.toFixed(1)} / 示范 ${d.demo_value.toFixed(1)}，差 ${d.delta.toFixed(1)}</p>`).join("")}
      <p><button class="danger" id="del">删除误切</button></p>`;
  }

  document.querySelectorAll(".stroke-item").forEach(el => {
    el.onclick = () => {
      window._sel = el.dataset.id;
      const s = strokes.find(x => x.stroke.stroke_id === window._sel);
      const v = $("#v");
      if (s && v) v.currentTime = s.stroke.start_s;
      renderJob(job);
      bindJobEvents(job);
    };
  });
  bindJobEvents(job, cur);
}

function bindJobEvents(job, cur) {
  const retry = $("#retry");
  if (retry) retry.onclick = async () => {
    await api(`/jobs/${job.job_id}/retry`, { method: "POST" });
    pollJob(job.job_id);
  };
  const type = $("#type");
  if (type && cur) type.onchange = async () => {
    const body = await api(`/jobs/${job.job_id}/strokes/${cur.stroke.stroke_id}/type`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stroke_type: type.value }),
    });
    window._job = body;
    renderJob(body);
  };
  const del = $("#del");
  if (del && cur) del.onclick = async () => {
    const body = await api(`/jobs/${job.job_id}/strokes/${cur.stroke.stroke_id}`, { method: "DELETE" });
    window._sel = null;
    renderJob(body);
  };
  const person = $("#person");
  if (person) person.onchange = async () => {
    const body = await api(`/jobs/${job.job_id}/person`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ person_id: Number(person.value) }),
    });
    renderJob(body);
  };
  const v = $("#v");
  const c = $("#c");
  const loopBtn = $("#loop");
  let looping = false;
  if (loopBtn && cur && v) {
    loopBtn.onclick = () => {
      looping = !looping;
      v.currentTime = cur.stroke.start_s;
      v.play();
    };
    v.ontimeupdate = () => {
      if (looping && v.currentTime > cur.stroke.end_s) v.currentTime = cur.stroke.start_s;
    };
  }
  if (v && c) {
    let last = -1;
    const tick = async () => {
      if (!v.duration) return;
      const t = v.currentTime;
      const framesGuess = 30;
      const frame = Math.round(t * framesGuess);
      if (frame === last) return;
      last = frame;
      try {
        const poses = await api(`/jobs/${job.job_id}/poses?from_frame=${Math.max(0, frame - 1)}&to_frame=${frame + 1}`);
        const hit = poses.find(p => p.frame_index === frame) || poses[0];
        const pid = job.person_id;
        const people = hit ? hit.people.filter(p => p.person_id === pid) : [];
        drawPose(c, v, people.length ? people : (hit ? hit.people : []));
      } catch (_) { /* ignore */ }
    };
    v.addEventListener("timeupdate", tick);
    v.addEventListener("seeked", tick);
  }
  fetch(`/jobs/${job.job_id}/demo-video`).then(r => {
    if (!r.ok) return;
    const slot = $("#demo-slot");
    if (!slot || !cur) return;
    slot.innerHTML = `<p class="muted">示范视频</p><video id="dv" src="/jobs/${job.job_id}/demo-video" controls></video>`;
    const dv = $("#dv");
    const pv = $("#v");
    if (dv && pv && cur.demo_diffs) {
      pv.addEventListener("seeked", () => {
        dv.currentTime = cur.stroke.impact_s;
      });
    }
  });
}

async function pollJob(id) {
  const job = await api(`/jobs/${id}`);
  window._job = job;
  renderJob(job);
  if (["queued", "extracting_pose", "segmenting", "scoring"].includes(job.status)) {
    setTimeout(() => pollJob(id), 800);
  }
}

async function route() {
  const id = jobIdFromHash();
  if (!id) {
    const jobs = await api("/jobs");
    renderHome(jobs);
    return;
  }
  await pollJob(id);
}

window.addEventListener("hashchange", route);
route().catch((err) => {
  document.getElementById("app").innerHTML = `<p class="error">${err.message}</p>`;
});
