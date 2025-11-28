document.getElementById("pipelineForm").addEventListener("submit", async (e) => {
    e.preventDefault();
  
    const form = e.target;
    const data = new FormData(form);
  
    // simple UI feedback
    const resultArea = document.getElementById("resultArea");
    const resultContent = document.getElementById("resultContent");
    const errorContent = document.getElementById("errorContent");
    resultArea.style.display = "block";
    resultContent.innerHTML = "<p>Running pipeline (this may take a while)...</p>";
    errorContent.innerText = "";
  
    try {
      const resp = await fetch("/run_pipeline", {
        method: "POST",
        body: data
      });
  
      const json = await resp.json();
      if (!resp.ok) {
        resultContent.innerHTML = "";
        errorContent.innerText = `Error: ${json.error || JSON.stringify(json)}`;
        if (json.traceback) {
          const pre = document.createElement("pre");
          pre.innerText = json.traceback;
          resultContent.appendChild(pre);
        }
        return;
      }
  
      // display preview and links if available
      resultContent.innerHTML = `<pre>${JSON.stringify(json.result_preview, null, 2)}</pre>`;
  
      if (json.artifacts_dir) {
        const artifacts = json.artifacts_dir.split("/").pop();
        resultContent.innerHTML += `<p>Artifacts saved to: <strong>${json.artifacts_dir}</strong></p>`;
        resultContent.innerHTML += `<p>Download result JSON: <a href="/artifacts/${artifacts}/result.json" target="_blank">result.json</a></p>`;
      }
  
    } catch (err) {
      resultContent.innerHTML = "";
      errorContent.innerText = "Unexpected error running pipeline: " + err.toString();
    }
  });
  