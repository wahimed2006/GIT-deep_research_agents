/* Build a self-contained HTML report from evaluation-results.json. */
const fs = require("fs");
const path = process.argv[2] || "evaluation-results.json";
const output = process.argv[3] || "evaluation-report.html";
const data = JSON.parse(fs.readFileSync(path, "utf8"));

const rows = data.results.map((entry) => {
  const stages = (entry.telemetry && entry.telemetry.stages) || {};
  return { ...entry, stages, total: stages.total || 0 };
});
const average = (name) => {
  const values = rows.map((row) => row.stages[name] || 0).filter(Boolean);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
};
const stageNames = ["decomposition", "search", "analysis", "scraping", "synthesis"];
const tableRows = rows.map((row) => `
  <tr><td>${row.position}</td><td>${escapeHtml(row.query)}</td><td>${row.total.toFixed(3)} s</td>
  <td>${(row.telemetry.questions_found || []).length}</td><td>${row.metadata?.scraped_sources || 0}</td>
  <td>${escapeHtml(row.error || "OK")}</td></tr>`).join("");
const chartLabels = JSON.stringify(rows.map((row) => `Q${row.position}`));
const chartValues = JSON.stringify(rows.map((row) => row.total));
const stageSummary = stageNames.map((name) => `<li><strong>${name}</strong>: ${average(name).toFixed(3)} s en moyenne</li>`).join("");

const html = `<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Agent evaluation report</title>
<style>body{font:14px system-ui;margin:2rem;color:#17202a}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd;padding:.45rem;text-align:left}th{background:#e7eef5}canvas{width:100%;height:280px;border:1px solid #ccd}li{display:inline-block;margin-right:1.5rem}</style></head>
<body><h1>Rapport d'évaluation des agents</h1><p>${rows.length} requêtes évaluées.</p>
<ul>${stageSummary}</ul><canvas id="duration" width="1200" height="280"></canvas>
<table><thead><tr><th>#</th><th>Question</th><th>Durée totale</th><th>Questions trouvées</th><th>Sources scrapées</th><th>Statut</th></tr></thead><tbody>${tableRows}</tbody></table>
<script>
const labels=${chartLabels}; const values=${chartValues}; const canvas=document.getElementById("duration"); const ctx=canvas.getContext("2d");
const pad=42, width=canvas.width-pad*2, height=canvas.height-pad*2, max=Math.max(...values, 1);
ctx.strokeStyle="#6b7c93"; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+height); ctx.lineTo(pad+width,pad+height); ctx.stroke();
ctx.strokeStyle="#1d7a8c"; ctx.lineWidth=2; ctx.beginPath(); values.forEach((value,index)=>{const x=pad+index*width/Math.max(values.length-1,1), y=pad+height-value/max*height; index?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
ctx.fillStyle="#17202a"; values.forEach((value,index)=>{const x=pad+index*width/Math.max(values.length-1,1), y=pad+height-value/max*height; ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill(); if(index%Math.max(1,Math.floor(values.length/10))===0)ctx.fillText(labels[index],x-10,pad+height+18);});
</script></body></html>`;
fs.writeFileSync(output, html);
console.log(`Report written to ${output}`);

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char])); }
