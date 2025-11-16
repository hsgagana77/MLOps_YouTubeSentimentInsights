async function ingestYoutube(){
  const form = document.getElementById("ytForm");
  const fd = new FormData(form);
  const body = {
    video_id: fd.get("video_id"),
    api_key: fd.get("api_key")
  };
  const res = await fetch('/ingest_youtube', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  const data = await res.json();
  const div = document.getElementById("ytResults");
  if(data.error){
    div.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
  } else {
    let html = '<ul class="list-group">';
    data.results.forEach(r => {
      const color = r.pred==1 ? 'success' : (r.pred==0 ? 'secondary' : 'danger');
      html += `<li class="list-group-item"><strong><span class="badge bg-${color}">${r.label}</span></strong> ${r.comment}</li>`;
    });
    html += '</ul>';
    div.innerHTML = html;
  }
}

async function generateWordcloud(){
  const ta = document.querySelector("#wcForm textarea");
  const comments = ta.value.split("\n").filter(Boolean);
  const res = await fetch('/generate_wordcloud', {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({comments})
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  document.getElementById("wcImg").src = url;
}

async function generatePie(){
  const counts = {
    '1': document.getElementById("pc_1").value || 0,
    '0': document.getElementById("pc_0").value || 0,
    '-1': document.getElementById("pc_-1").value || 0,
  };
  const res = await fetch('/generate_chart', {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({sentiment_counts: counts})
  });
  const blob = await res.blob();
  document.getElementById("pcImg").src = URL.createObjectURL(blob);
}