const analyseBtn   = document.getElementById("analyseBtn");
const askBtn       = document.getElementById("askBtn");
const qInput       = document.getElementById("questionInput");
const statusBox    = document.getElementById("status");
const answerBox    = document.getElementById("answerBox");

// helper
function setStatus(txt){ statusBox.textContent = txt; }
function showAnswer(html){
    answerBox.innerHTML = html;
    answerBox.style.display = "block";
}

function ensureContentScript(tabId, cb){
    chrome.scripting.executeScript({target:{tabId},files:["contentScript.js"]})
          .then(cb)
          .catch(err=>setStatus("Inject error: "+err.message));
}

// Analyse page
analyseBtn.onclick = ()=>{
  setStatus("Analysing…");
  answerBox.style.display="none";
  chrome.tabs.query({active:true,currentWindow:true},([tab])=>{
    if(!tab){ setStatus("No active tab"); return;}
    ensureContentScript(tab.id, ()=>{
      chrome.tabs.sendMessage(tab.id,{action:"analyze"},res=>{
        if(chrome.runtime.lastError){ setStatus(chrome.runtime.lastError.message); return;}
        if(res?.ok){
          const d = res.data;
          const riskCls = d.risk_level.toLowerCase();
          showAnswer(
            `<strong>📋 Risk Summary</strong><br>
             Type: ${d.document_type}<br>
             Risk Score: ${d.risk_score}/100<br>
             Risk Level: <span class="risk ${riskCls}">${d.risk_level}</span><br><br>
             ${d.summary}`
          );
          setStatus("Analysis complete.");
        }else{
          setStatus(res?.error||"Analysis failed");
        }
      });
    });
  });
};

// Ask question
askBtn.onclick = ()=> sendQuestion(qInput.value.trim());

// Example buttons
document.querySelectorAll("#examples button").forEach(btn=>{
  btn.onclick = ()=> sendQuestion(btn.dataset.q);
});

function sendQuestion(question){
  if(!question){ setStatus("Type a question."); return;}
  setStatus("Getting answer…");
  answerBox.style.display="none";
  chrome.tabs.query({active:true,currentWindow:true},([tab])=>{
    if(!tab){ setStatus("No active tab"); return;}
    ensureContentScript(tab.id, ()=>{
      chrome.tabs.sendMessage(tab.id,{action:"ask",question},res=>{
        if(chrome.runtime.lastError){ setStatus(chrome.runtime.lastError.message); return;}
        if(res?.ok){
          const d = res.data;
          showAnswer(
            `<strong>💡 Answer</strong><br>${d.answer}<br><br>
             Confidence: ${(d.confidence*100).toFixed(1)}%<br>
             Relevant Chunks: ${d.relevant_chunks}<br>
             Method: ${d.method}`
          );
          setStatus("Answer ready.");
        }else{
          setStatus(res?.error||"Question failed");
        }
      });
    });
  });
}
