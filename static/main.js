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
      // If server returns an error
      if (!resp.ok) {
        const errorText = await resp.text();
        resultContent.textContent = "";
        errorContent.textContent = `Error:\n${errorText}`;
        return;
      }

      // Read pipeline output as text
      const resultText = await resp.text();
      resultContent.textContent = resultText;

      // Provide a download link for result.txt
      // Create a Blob and Object URL
      const blob = new Blob([resultText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);

      // Remove old link if exists
      let oldLink = document.getElementById("downloadLink");
      if (oldLink) oldLink.remove();

      const link = document.createElement("a");
      link.id = "downloadLink";
      link.href = url;
      link.download = "result.txt";
      link.textContent = "Download result.txt";
      link.style.display = "block";
      link.style.marginTop = "10px";
      resultArea.appendChild(link);
  
      // const json = await resp.json();
      // if (!resp.ok) {
      //   resultContent.innerHTML = "";
      //   errorContent.innerText = `Error: ${json.error || JSON.stringify(json)}`;
      //   if (json.traceback) {
      //     const pre = document.createElement("pre");
      //     pre.innerText = json.traceback;
      //     resultContent.appendChild(pre);
      //   }
      //   return;
      // }
  
      // // display preview and links if available
      // resultContent.innerHTML = `<pre>${JSON.stringify(json.result_preview, null, 2)}</pre>`;
  
      // if (json.artifacts_dir) {
      //   const artifacts = json.artifacts_dir.split("/").pop();
      //   resultContent.innerHTML += `<p>Artifacts saved to: <strong>${json.artifacts_dir}</strong></p>`;
      //   resultContent.innerHTML += `<p>Download result text: <a href="/artifacts/${artifacts}/result.txt" target="_blank">result.txt</a></p>`;
      // }
  
    } catch (err) {
      resultContent.innerHTML = "";
      errorContent.innerText = "Unexpected error running pipeline: " + err.toString();
    }
  });
  
