// Extract up to 5 000 chars of probable legal text
function extractLegalText(){
  const selectors = [".terms",".terms-of-service",".privacy-policy",".legal",".tos",".privacy","article","section","#legal"];
  const pieces = [];
  selectors.forEach(sel=>{
    document.querySelectorAll(sel).forEach(el=>{
      if(el.innerText && el.innerText.length>100) pieces.push(el.innerText);
    });
  });
  if(!pieces.length) pieces.push(document.body.innerText);
  return pieces.join("\n\n").slice(0,5000);
}

chrome.runtime.onMessage.addListener((msg,sender,sendResponse)=>{
  if(msg.action==="analyze"){
    doAnalyse(sendResponse);         // async
    return true;                     // keep channel open
  }
  if(msg.action==="ask"){
    doAsk(msg.question,sendResponse);// async
    return true;
  }
});

/* ---------- helpers ---------- */
async function doAnalyse(sendResponse){
  const text = extractLegalText();
  if(text.length<100){ sendResponse({error:"Text too short"}); return;}
  try{
    const res = await fetch("http://localhost:8000/analyze-document/",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text})
    });
    if(!res.ok){
      const detail = await res.text();
      sendResponse({error:`Backend ${res.status}: ${detail}`});
      return;
    }
    const data = await res.json();
    sendResponse({ok:true,data});
  }catch(err){
    sendResponse({error:"Network error "+err.message});
  }
}

async function doAsk(question,sendResponse){
  const text = extractLegalText();
  if(text.length<100){ sendResponse({error:"Text too short"}); return;}
  try{
    const res = await fetch("http://localhost:8000/ask-screen/",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question,text})
    });
    if(!res.ok){
      const detail = await res.text();
      sendResponse({error:`Backend ${res.status}: ${detail}`});
      return;
    }
    const data = await res.json();
    sendResponse({ok:true,data});
  }catch(err){
    sendResponse({error:"Network error "+err.message});
  }
}
